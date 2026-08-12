#!/usr/bin/env python
"""Collect compact stage-one metadata and metrics from external run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ROOT = Path("/media/zfei/Glowat512/projects/lerobot-arch-exp")
RUN_RE = re.compile(r"^(?P<variant>.+)_seed(?P<seed>\d+)_(?P<steps>\d+)steps$")
LEARNABLE_PARAM_RE = re.compile(r"num_learnable_params=(?P<params>\d+)")
TOTAL_PARAM_RE = re.compile(r"num_total_params=(?P<params>\d+)")
UPDATE_TIME_RE = re.compile(r"updt_s:(?P<seconds>[0-9.]+)")
VALIDATION_RE = re.compile(r"Validation at step (?P<step>\d+): (?P<metrics>[^\r\n]+)")
WRAPPER_TIME_RE = re.compile(
    r"\[(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (?P<event>starting|completed)"
)
COMPARISON_METRICS = (
    "rotation_chunk_mean_deg",
    "rotation_end_deg",
    "xyz_chunk_mean_m",
    "xyz_end_m",
    "gripper_chunk_mean",
    "gripper_end",
    "rot_jerk_deg",
    "xyz_jerk_m",
)
EXTRA_PAIRED_VARIANTS = (
    ("act_r50_large", "act_r50_vae"),
    ("act_r18_flow_u_lr1e5", "act_r18_l1"),
    ("act_r18_flow_u_lr1e4", "act_r18_flow_u_lr1e5"),
    ("act_r18_flow_beta_lr1e4", "act_r18_flow_u_lr1e4"),
    ("diffusion_r18", "act_r18_l1"),
    ("umi_official_dp", "diffusion_r18"),
    ("umi_official_dp", "act_r18_l1"),
    ("umi_official_transformer_dp", "umi_official_dp"),
    ("umi_official_transformer_dp", "act_r18_l1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--eval_dir_name",
        default="eval_common_h32",
        help="Evaluation namespace to collect (default: fixed common-horizon reports).",
    )
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
        return {
            "status": "missing_log",
            "parameters": None,
            "learnable_parameters": None,
            "total_parameters": None,
            "wall_seconds": None,
            "validation": [],
        }
    text = log_path.read_text(errors="replace")
    learnable_parameter_match = LEARNABLE_PARAM_RE.search(text)
    total_parameter_match = TOTAL_PARAM_RE.search(text)
    learnable_parameters = (
        int(learnable_parameter_match["params"]) if learnable_parameter_match else None
    )
    total_parameters = int(total_parameter_match["params"]) if total_parameter_match else None
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
    status = "complete" if "completed" in times else "running_or_failed"
    if status != "complete" and "End of training" in text:
        status = "training_finished_without_wrapper_marker"
    return {
        "status": status,
        # Keep `parameters` as a compatibility alias for the online model size.
        # EMA policies register a second frozen copy, so `num_total_params` is
        # checkpoint/training state rather than inference architecture size.
        "parameters": learnable_parameters if learnable_parameters is not None else total_parameters,
        "learnable_parameters": learnable_parameters,
        "total_parameters": total_parameters,
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
        "query_min_action_offset": report.get("query_action_offset_bounds", {}).get("min"),
        "query_max_action_offset": report.get("query_action_offset_bounds", {}).get("max"),
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


def aggregate_episode_metrics(
    reports: list[dict[str, Any]],
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    """Average inference seeds within each episode, preserving exact episode pairing."""
    if not reports:
        raise ValueError("At least one evaluation report is required")
    episode_id_sets = [set(report["summary"]["per_episode"]) for report in reports]
    if any(episode_ids != episode_id_sets[0] for episode_ids in episode_id_sets[1:]):
        raise ValueError("Inference-seed evaluation reports have mismatched episode IDs")
    episode_ids = tuple(sorted(episode_id_sets[0]))
    if not episode_ids:
        raise ValueError("Evaluation reports have no common episodes")
    return episode_ids, {
        metric: np.asarray(
            [
                statistics.mean(
                    float(report["summary"]["per_episode"][episode_id][metric]) for report in reports
                )
                for episode_id in episode_ids
            ],
            dtype=np.float64,
        )
        for metric in COMPARISON_METRICS
    }


def bootstrap_mean_interval(
    values: np.ndarray, *, rng: np.random.Generator, num_resamples: int = 10_000
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(num_resamples, len(values)))
    resampled_means = values[indices].mean(axis=1)
    low, high = np.percentile(resampled_means, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_paired_improvement_interval(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    rng: np.random.Generator,
    num_resamples: int = 10_000,
) -> tuple[float, float]:
    indices = rng.integers(0, len(baseline), size=(num_resamples, len(baseline)))
    baseline_means = baseline[indices].mean(axis=1)
    candidate_means = candidate[indices].mean(axis=1)
    improvements = (baseline_means - candidate_means) / baseline_means * 100
    low, high = np.percentile(improvements, [2.5, 97.5])
    return float(low), float(high)


def summarize_variants(
    reports_by_run: dict[str, list[dict[str, Any]]], runs_by_name: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return seed-averaged summaries and paired differences from fresh ACT R18."""
    episode_data_by_run = {
        run_name: aggregate_episode_metrics(reports) for run_name, reports in reports_by_run.items()
    }
    summary_rows = []
    for run_name, (_, metrics) in episode_data_by_run.items():
        reports = reports_by_run[run_name]
        run = runs_by_name[run_name]
        row = {
            **run,
            "num_inference_seeds": len(reports),
            "inference_median_seconds": statistics.mean(
                float(report["inference_latency_seconds"]["median_seconds"]) for report in reports
            ),
            "inference_p95_seconds": statistics.mean(
                float(report["inference_latency_seconds"]["p95_seconds"]) for report in reports
            ),
            "cuda_peak_memory_bytes": max(int(report["cuda_peak_memory_bytes"]) for report in reports),
        }
        rng = np.random.default_rng(0)
        for metric, values in metrics.items():
            low, high = bootstrap_mean_interval(values, rng=rng)
            seed_means = [float(report["summary"]["episode_balanced"][metric]) for report in reports]
            row[metric] = float(values.mean())
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_inference_seed_sd"] = float(np.std(seed_means))
        summary_rows.append(row)

    comparison_pairs = set()
    for run_name, run in runs_by_name.items():
        baseline_name = f"act_r18_vae_seed{run['training_seed']}_{run['steps']}steps"
        if run_name != baseline_name and baseline_name in episode_data_by_run:
            comparison_pairs.add((run_name, baseline_name))
    for candidate_variant, baseline_variant in EXTRA_PAIRED_VARIANTS:
        for run_name, run in runs_by_name.items():
            if run["variant"] != candidate_variant:
                continue
            baseline_name = f"{baseline_variant}_seed{run['training_seed']}_{run['steps']}steps"
            if baseline_name in episode_data_by_run:
                comparison_pairs.add((run_name, baseline_name))

    comparison_rows = []
    for run_name, baseline_name in sorted(comparison_pairs):
        candidate_episode_ids, candidate_metrics = episode_data_by_run[run_name]
        run = runs_by_name[run_name]
        baseline_episode_ids, baseline_metrics = episode_data_by_run[baseline_name]
        if candidate_episode_ids != baseline_episode_ids:
            raise ValueError(f"Episode ID mismatch between {run_name} and {baseline_name}")
        for metric in COMPARISON_METRICS:
            baseline = baseline_metrics[metric]
            candidate = candidate_metrics[metric]
            differences = candidate - baseline
            rng = np.random.default_rng(0)
            diff_low, diff_high = bootstrap_mean_interval(differences, rng=rng)
            rng = np.random.default_rng(0)
            improvement_low, improvement_high = bootstrap_paired_improvement_interval(
                baseline, candidate, rng=rng
            )
            comparison_rows.append(
                {
                    **run,
                    "baseline_run_name": baseline_name,
                    "baseline_variant": runs_by_name[baseline_name]["variant"],
                    "metric": metric,
                    "candidate_mean": float(candidate.mean()),
                    "baseline_mean": float(baseline.mean()),
                    "paired_difference": float(differences.mean()),
                    "paired_difference_ci_low": diff_low,
                    "paired_difference_ci_high": diff_high,
                    "paired_improvement_percent": float(
                        (baseline.mean() - candidate.mean()) / baseline.mean() * 100
                    ),
                    "paired_improvement_percent_ci_low": improvement_low,
                    "paired_improvement_percent_ci_high": improvement_high,
                }
            )
    return summary_rows, comparison_rows


def summarize_training_seed_variability(
    variant_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate independent training runs without conflating them with inference seeds."""
    variants: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in variant_rows:
        variants[(row["variant"], row["steps"])].append(row)

    variant_seed_rows = []
    for (variant, steps), rows in sorted(variants.items()):
        seeds = sorted(int(row["training_seed"]) for row in rows)
        result: dict[str, Any] = {
            "variant": variant,
            "steps": steps,
            "num_training_seeds": len(seeds),
            "training_seeds": json.dumps(seeds),
        }
        for metric in COMPARISON_METRICS:
            values = [float(row[metric]) for row in rows]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_training_seed_sd"] = statistics.stdev(values) if len(values) > 1 else None
            result[f"{metric}_min"] = min(values)
            result[f"{metric}_max"] = max(values)
        variant_seed_rows.append(result)

    comparisons: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        comparisons[
            (row["variant"], row["baseline_variant"], row["steps"], row["metric"])
        ].append(row)

    comparison_seed_rows = []
    for (variant, baseline, steps, metric), rows in sorted(comparisons.items()):
        seeds = sorted(int(row["training_seed"]) for row in rows)
        differences = [float(row["paired_difference"]) for row in rows]
        improvements = [float(row["paired_improvement_percent"]) for row in rows]
        comparison_seed_rows.append(
            {
                "variant": variant,
                "baseline_variant": baseline,
                "steps": steps,
                "metric": metric,
                "num_training_seeds": len(seeds),
                "training_seeds": json.dumps(seeds),
                "paired_difference_mean": statistics.mean(differences),
                "paired_difference_training_seed_sd": (
                    statistics.stdev(differences) if len(differences) > 1 else None
                ),
                "paired_difference_min": min(differences),
                "paired_difference_max": max(differences),
                "paired_improvement_percent_mean": statistics.mean(improvements),
                "paired_improvement_percent_training_seed_sd": (
                    statistics.stdev(improvements) if len(improvements) > 1 else None
                ),
                "paired_improvement_percent_min": min(improvements),
                "paired_improvement_percent_max": max(improvements),
            }
        )
    return variant_seed_rows, comparison_seed_rows


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
    reports_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    runs_by_name = {}
    for run_dir in sorted((args.artifact_root / "train").glob("*")):
        if not run_dir.is_dir() or RUN_RE.fullmatch(run_dir.name) is None:
            continue
        run = parse_run_name(run_dir.name)
        runs_by_name[run_dir.name] = run
        log = parse_log(args.artifact_root / "logs" / f"{run_dir.name}.log")
        train_rows.append({**run, **{key: value for key, value in log.items() if key != "validation"}})
        validation_rows.extend({**run, **metrics} for metrics in log["validation"])
        for report_path in sorted(
            (args.artifact_root / args.eval_dir_name / run_dir.name).glob(
                "seed*/*_open_loop_metrics.json"
            )
        ):
            evaluation_rows.append(flatten_evaluation(report_path, run))
            report = json.loads(report_path.read_text())
            bounds = report.get("query_action_offset_bounds")
            if args.eval_dir_name == "eval_common_h32" and bounds != {"min": -1, "max": 31}:
                raise ValueError(f"Unexpected common-horizon query bounds in {report_path}: {bounds}")
            reports_by_run[run_dir.name].append(report)

    variant_rows, comparison_rows = summarize_variants(reports_by_run, runs_by_name)
    variant_seed_rows, comparison_seed_rows = summarize_training_seed_variability(
        variant_rows, comparison_rows
    )

    result_dir = args.artifact_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    write_csv(result_dir / "stage1_runs.csv", train_rows)
    write_csv(result_dir / "stage1_validation.csv", validation_rows)
    write_csv(result_dir / "stage1_evaluations.csv", evaluation_rows)
    write_csv(result_dir / "stage1_variant_summary.csv", variant_rows)
    write_csv(result_dir / "stage1_paired_comparisons.csv", comparison_rows)
    write_csv(result_dir / "training_seed_variant_summary.csv", variant_seed_rows)
    write_csv(result_dir / "training_seed_paired_comparisons.csv", comparison_seed_rows)
    write_csv(
        result_dir / "stage1_paired_vs_act_r18.csv",
        [row for row in comparison_rows if row["baseline_variant"] == "act_r18_vae"],
    )
    (result_dir / "stage1_results.json").write_text(
        json.dumps(
            {
                "runs": train_rows,
                "validation": validation_rows,
                "evaluations": evaluation_rows,
                "variant_summary": variant_rows,
                "paired_comparisons": comparison_rows,
                "training_seed_variant_summary": variant_seed_rows,
                "training_seed_paired_comparisons": comparison_seed_rows,
            },
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

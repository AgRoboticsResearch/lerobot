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

DEFAULT_ROOT = Path("/mnt/data1/projects/lerobot-arch-exp")
RUN_RE = re.compile(r"^(?P<variant>.+)_seed(?P<seed>\d+)_(?P<steps>\d+)steps$")
# The historical production run (§9.2.7) predates the seed/budget naming
# convention: act_umi_identity_rot6d_1459_<7-digit-step>steps, training seed
# 1000 (verified in §3's audit of its saved train_config.json).
HIST_RUN_RE = re.compile(r"^(?P<variant>act_umi_identity_rot6d_1459)_(?P<steps>\d{7})steps$")
REPORT_STEP_RE = re.compile(r"_(?P<step>\d{6})_open_loop_metrics\.json$")
# v2-metric reports may carry 6- or 7-digit checkpoint steps and the
# historical family omits the run-name prefix; evaluated_step is taken from
# the report filename, which is authoritative over the directory budget
# (early-stopped companion runs are evaluated at their true final step).
V2_REPORT_STEP_RE = re.compile(r"_(?P<step>\d{6,7})_open_loop_metrics\.json$")
SEED_DIR_RE = re.compile(r"^seed(?P<seed>\d+)$")
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
    ("act_r50_vae", "act_r50_v1_vae"),
    ("act_r50_large", "act_r50_vae"),
    ("act_r18_flow_u_lr1e5", "act_r18_l1"),
    ("act_r18_diffusion_lr1e5", "act_r18_l1"),
    ("act_r18_diffusion_lr1e5", "act_r18_flow_u_lr1e5"),
    ("act_r18_flow_u_lr1e4", "act_r18_flow_u_lr1e5"),
    ("act_r18_flow_beta_lr1e4", "act_r18_flow_u_lr1e4"),
    ("diffusion_r18", "act_r18_l1"),
    ("diffusion_r18", "act_r18_diffusion_lr1e5"),
    ("umi_official_dp", "diffusion_r18"),
    ("umi_official_dp", "act_r18_l1"),
    ("umi_official_transformer_dp", "umi_official_dp"),
    ("umi_official_transformer_dp", "act_r18_l1"),
    ("smolvla_axis_angle", "smolvla_rot6d"),
    ("lingbot_va_axis_angle", "smolvla_axis_angle"),
)
# v2 metric set (2026-08-16/17 evaluator extensions): per-component L1,
# per-dimension MSE, and pi0.5-style thresholded accuracy at tau=0.5/0.1
# (plus component views). Present in reports produced after those commits;
# older reports (e.g. the kiwi pi0.5-port JSONs) predate them.
V2_METRICS = (
    "xyz_l1_per_dim_m",
    "xyz_mse_per_dim_m2",
    "rotvec_l1_per_dim_deg",
    "rotvec_mse_per_dim_deg2",
    "action_acc_at_0p5",
    "action_acc_at_0p1",
    "xyz_acc_at_0p5",
    "xyz_acc_at_0p1",
    "rotvec_acc_at_0p5",
    "rotvec_acc_at_0p1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--eval_dir_name",
        default="eval_common_h32",
        help="Evaluation namespace to collect (default: fixed common-horizon reports).",
    )
    parser.add_argument("--expected_num_episodes", type=int, default=100)
    parser.add_argument("--expected_samples_per_episode", type=int, default=5)
    parser.add_argument(
        "--v2_eval_roots",
        default=None,
        help=(
            "Comma-separated evaluation roots to scan with the v2-metric pass "
            "(e.g. <artifact_root>/reeval_v2metrics/eval_common_h32). Runs the v2 "
            "pass INSTEAD of the strict matrix scan: the shadow tree holds "
            "early-stopped companions evaluated below their directory budget and "
            "historical runs outside the seed/budget naming convention, both of "
            "which the strict pass rejects by design."
        ),
    )
    parser.add_argument(
        "--v2_out_dir",
        type=Path,
        default=None,
        help="Output directory for the v2 pass (default: <first v2 root>/../results).",
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


def validate_report_checkpoint_step(report_path: Path, expected_steps: int) -> None:
    match = REPORT_STEP_RE.search(report_path.name)
    if match is None or int(match["step"]) != expected_steps:
        raise ValueError(
            f"Evaluation checkpoint step does not match run budget for {report_path}: "
            f"expected {expected_steps}"
        )


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
    learnable_parameters = int(learnable_parameter_match["params"]) if learnable_parameter_match else None
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


def validate_evaluation_report(
    report_path: Path,
    report: dict[str, Any],
    *,
    expected_num_episodes: int,
    expected_samples_per_episode: int,
) -> int:
    """Validate fixed-query provenance and return the recorded inference seed."""
    seed_match = SEED_DIR_RE.fullmatch(report_path.parent.name)
    if seed_match is None:
        raise ValueError(f"Unexpected evaluation seed directory: {report_path.parent}")
    directory_seed = int(seed_match["seed"])
    report_seed = int(report["seed"])
    if report_seed != directory_seed:
        raise ValueError(
            f"Inference seed mismatch in {report_path}: directory={directory_seed}, report={report_seed}"
        )

    summary = report["summary"]
    per_episode = summary["per_episode"]
    reported_num_episodes = int(summary["num_episodes"])
    if reported_num_episodes != expected_num_episodes or len(per_episode) != expected_num_episodes:
        raise ValueError(
            f"Unexpected episode count in {report_path}: "
            f"summary={reported_num_episodes}, per_episode={len(per_episode)}, "
            f"expected={expected_num_episodes}"
        )
    expected_num_samples = expected_num_episodes * expected_samples_per_episode
    if int(summary["num_samples"]) != expected_num_samples:
        raise ValueError(
            f"Unexpected query count in {report_path}: "
            f"got={summary['num_samples']}, expected={expected_num_samples}"
        )
    return report_seed


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


def hierarchical_bootstrap_mean_interval(
    groups: list[np.ndarray], *, rng: np.random.Generator, num_resamples: int = 10_000
) -> tuple[float, float]:
    """Resample training runs first, then episodes within each selected run."""
    if not groups:
        raise ValueError("At least one training-seed group is required")
    if len(groups) == 1:
        return bootstrap_mean_interval(groups[0], rng=rng, num_resamples=num_resamples)
    if len({len(group) for group in groups}) != 1:
        raise ValueError("Training-seed groups have different episode counts")
    values = np.stack(groups)
    num_groups, num_episodes = values.shape
    group_indices = rng.integers(0, num_groups, size=(num_resamples, num_groups))
    episode_indices = rng.integers(0, num_episodes, size=(num_resamples, num_groups, num_episodes))
    resampled = values[group_indices[:, :, None], episode_indices]
    low, high = np.percentile(resampled.mean(axis=(1, 2)), [2.5, 97.5])
    return float(low), float(high)


def hierarchical_bootstrap_paired_improvement_interval(
    baseline_groups: list[np.ndarray],
    candidate_groups: list[np.ndarray],
    *,
    rng: np.random.Generator,
    num_resamples: int = 10_000,
) -> tuple[float, float]:
    """Hierarchically resample paired policies using identical seeds and episodes."""
    if len(baseline_groups) != len(candidate_groups) or not baseline_groups:
        raise ValueError("Paired policies must have the same nonzero number of training seeds")
    if len(baseline_groups) == 1:
        return bootstrap_paired_improvement_interval(
            baseline_groups[0], candidate_groups[0], rng=rng, num_resamples=num_resamples
        )
    shapes = {group.shape for group in baseline_groups + candidate_groups}
    if len(shapes) != 1:
        raise ValueError("Paired training-seed groups have different episode shapes")
    baselines = np.stack(baseline_groups)
    candidates = np.stack(candidate_groups)
    num_groups, num_episodes = baselines.shape
    group_indices = rng.integers(0, num_groups, size=(num_resamples, num_groups))
    episode_indices = rng.integers(0, num_episodes, size=(num_resamples, num_groups, num_episodes))
    resampled_baselines = baselines[group_indices[:, :, None], episode_indices]
    resampled_candidates = candidates[group_indices[:, :, None], episode_indices]
    baseline_means = resampled_baselines.mean(axis=(1, 2))
    candidate_means = resampled_candidates.mean(axis=(1, 2))
    improvements = (baseline_means - candidate_means) / baseline_means * 100
    low, high = np.percentile(improvements, [2.5, 97.5])
    return float(low), float(high)


def summarize_variants(
    reports_by_run: dict[str, list[dict[str, Any]]], runs_by_name: dict[str, dict[str, Any]]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, tuple[tuple[str, ...], dict[str, np.ndarray]]],
]:
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
        for metric, values in metrics.items():
            low, high = bootstrap_mean_interval(values, rng=np.random.default_rng(0))
            seed_means = [float(report["summary"]["episode_balanced"][metric]) for report in reports]
            row[metric] = float(values.mean())
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_inference_seed_sd"] = float(np.std(seed_means))
        summary_rows.append(row)

    comparison_pairs = set()
    for run_name, run in runs_by_name.items():
        # Training directories appear before their decoded evaluations finish.
        # Never promote an unevaluated live/incomplete run into a paired
        # comparison merely because its baseline already has episode data.
        if run_name not in episode_data_by_run:
            continue
        baseline_name = f"act_r18_vae_seed{run['training_seed']}_{run['steps']}steps"
        if run_name != baseline_name and baseline_name in episode_data_by_run:
            comparison_pairs.add((run_name, baseline_name))
    for candidate_variant, baseline_variant in EXTRA_PAIRED_VARIANTS:
        for run_name, run in runs_by_name.items():
            if run["variant"] != candidate_variant or run_name not in episode_data_by_run:
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
    return summary_rows, comparison_rows, episode_data_by_run


def summarize_training_seed_variability(
    variant_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    episode_data_by_run: dict[str, tuple[tuple[str, ...], dict[str, np.ndarray]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate independent training runs without conflating them with inference seeds."""
    variants: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in variant_rows:
        variants[(row["variant"], row["steps"])].append(row)

    variant_seed_rows = []
    for (variant, steps), rows in sorted(variants.items()):
        seeds = sorted(int(row["training_seed"]) for row in rows)
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate training seed for {variant} at {steps} steps: {seeds}")
        result: dict[str, Any] = {
            "variant": variant,
            "steps": steps,
            "num_training_seeds": len(seeds),
            "training_seeds": json.dumps(seeds),
        }
        for metric in COMPARISON_METRICS:
            values = [float(row[metric]) for row in rows]
            episode_id_groups = [episode_data_by_run[row["run_name"]][0] for row in rows]
            if any(ids != episode_id_groups[0] for ids in episode_id_groups[1:]):
                raise ValueError(f"Episode ID mismatch across training seeds for {variant} at {steps}")
            metric_groups = [episode_data_by_run[row["run_name"]][1][metric] for row in rows]
            hierarchical_low, hierarchical_high = hierarchical_bootstrap_mean_interval(
                metric_groups, rng=np.random.default_rng(0)
            )
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_training_seed_sd"] = statistics.stdev(values) if len(values) > 1 else None
            result[f"{metric}_min"] = min(values)
            result[f"{metric}_max"] = max(values)
            result[f"{metric}_hierarchical_ci_low"] = hierarchical_low
            result[f"{metric}_hierarchical_ci_high"] = hierarchical_high
        variant_seed_rows.append(result)

    comparisons: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        comparisons[(row["variant"], row["baseline_variant"], row["steps"], row["metric"])].append(row)

    comparison_seed_rows = []
    for (variant, baseline, steps, metric), rows in sorted(comparisons.items()):
        seeds = sorted(int(row["training_seed"]) for row in rows)
        if len(seeds) != len(set(seeds)):
            raise ValueError(
                f"Duplicate paired training seed for {variant} vs {baseline} at {steps}: {seeds}"
            )
        baseline_seeds = sorted(
            int(parse_run_name(row["baseline_run_name"])["training_seed"]) for row in rows
        )
        if seeds != baseline_seeds:
            raise ValueError(
                f"Candidate/baseline training-seed mismatch for {variant} vs {baseline} "
                f"at {steps}: candidate={seeds}, baseline={baseline_seeds}"
            )
        differences = [float(row["paired_difference"]) for row in rows]
        improvements = [float(row["paired_improvement_percent"]) for row in rows]
        candidate_episode_ids = [episode_data_by_run[row["run_name"]][0] for row in rows]
        baseline_episode_ids = [episode_data_by_run[row["baseline_run_name"]][0] for row in rows]
        all_episode_ids = candidate_episode_ids + baseline_episode_ids
        if any(ids != all_episode_ids[0] for ids in all_episode_ids[1:]):
            raise ValueError(f"Episode ID mismatch across paired training seeds for {variant} vs {baseline}")
        candidate_groups = [episode_data_by_run[row["run_name"]][1][metric] for row in rows]
        baseline_groups = [episode_data_by_run[row["baseline_run_name"]][1][metric] for row in rows]
        difference_low, difference_high = hierarchical_bootstrap_mean_interval(
            [
                candidate - baseline_values
                for candidate, baseline_values in zip(candidate_groups, baseline_groups, strict=True)
            ],
            rng=np.random.default_rng(0),
        )
        improvement_low, improvement_high = hierarchical_bootstrap_paired_improvement_interval(
            baseline_groups, candidate_groups, rng=np.random.default_rng(0)
        )
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
                "paired_difference_hierarchical_ci_low": difference_low,
                "paired_difference_hierarchical_ci_high": difference_high,
                "paired_improvement_percent_mean": statistics.mean(improvements),
                "paired_improvement_percent_training_seed_sd": (
                    statistics.stdev(improvements) if len(improvements) > 1 else None
                ),
                "paired_improvement_percent_min": min(improvements),
                "paired_improvement_percent_max": max(improvements),
                "paired_improvement_percent_hierarchical_ci_low": improvement_low,
                "paired_improvement_percent_hierarchical_ci_high": improvement_high,
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


def parse_v2_run_name(name: str) -> dict[str, Any]:
    match = RUN_RE.fullmatch(name)
    if match is not None:
        values = match.groupdict()
        return {
            "run_name": name,
            "variant": values["variant"],
            "training_seed": int(values["seed"]),
            "budget_steps": int(values["steps"]),
        }
    match = HIST_RUN_RE.fullmatch(name)
    if match is not None:
        return {
            "run_name": name,
            "variant": match["variant"],
            "training_seed": 1000,
            "budget_steps": int(match["steps"]),
        }
    raise ValueError(f"Unrecognized v2 run name: {name}")


def aggregate_available_episode_metrics(
    reports: list[dict[str, Any]],
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    """Like aggregate_episode_metrics, but over every per-episode metric present
    in ALL reports of the run — v2 L1/MSE/accuracy@tau keys ride along when the
    reports carry them and are absent for pre-2026-08-16 reports."""
    if not reports:
        raise ValueError("At least one evaluation report is required")
    episode_id_sets = [set(report["summary"]["per_episode"]) for report in reports]
    if any(episode_ids != episode_id_sets[0] for episode_ids in episode_id_sets[1:]):
        raise ValueError("Inference-seed evaluation reports have mismatched episode IDs")
    episode_ids = tuple(sorted(episode_id_sets[0]))
    if not episode_ids:
        raise ValueError("Evaluation reports have no common episodes")
    available: set[str] = set(reports[0]["summary"]["per_episode"][episode_ids[0]])
    for report in reports[1:]:
        available &= set(report["summary"]["per_episode"][episode_ids[0]])
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
        for metric in sorted(available)
    }


def collect_v2_metrics(
    eval_roots: list[Path],
    out_dir: Path,
    *,
    expected_num_episodes: int,
    expected_samples_per_episode: int,
) -> None:
    """Collect the v2-metric (L1 / per-dim MSE / accuracy@tau) evaluation sweep.

    The strict matrix pass above keys runs off train/ directories and enforces
    report-step == budget-step. The v2 sweep lives in shadow eval roots whose
    runs violate both assumptions by design: early-stopped seed-23k companions
    (directory says 100000steps, real checkpoint 080000/050000) and the
    historical production curve (outside the naming convention). This pass
    therefore drives from the eval directories, records the authoritative
    evaluated step from each report filename, and skips unreadable report
    files (artifact-disk husks, §8 incidents 11-12) with a loud warning
    instead of aborting the sweep.
    """
    evaluation_rows = []
    run_rows = []
    reports_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    inference_seeds_by_run: dict[str, set[int]] = defaultdict(set)
    runs_by_name: dict[str, dict[str, Any]] = {}
    skipped: list[tuple[str, str]] = []
    for root in eval_roots:
        for run_dir in sorted(path for path in root.glob("*") if path.is_dir()):
            try:
                run = parse_v2_run_name(run_dir.name)
            except ValueError:
                continue
            if run_dir.name in runs_by_name:
                raise ValueError(f"Run directory appears in multiple v2 roots: {run_dir.name}")
            runs_by_name[run_dir.name] = run
            for report_path in sorted(run_dir.glob("seed*/*_open_loop_metrics.json")):
                try:
                    report = json.loads(report_path.read_text())
                except (json.JSONDecodeError, OSError) as error:
                    skipped.append((str(report_path), str(error)[:80]))
                    continue
                step_match = V2_REPORT_STEP_RE.search(report_path.name)
                if step_match is None:
                    raise ValueError(f"Cannot parse evaluated step from {report_path}")
                row = {
                    **run,
                    "evaluated_step": int(step_match["step"]),
                    "has_v2_metrics": all(
                        metric in report["summary"]["episode_balanced"] for metric in V2_METRICS
                    ),
                    **flatten_evaluation(report_path, run),
                }
                evaluation_rows.append(row)
                inference_seed = validate_evaluation_report(
                    report_path,
                    report,
                    expected_num_episodes=expected_num_episodes,
                    expected_samples_per_episode=expected_samples_per_episode,
                )
                if inference_seed in inference_seeds_by_run[run_dir.name]:
                    raise ValueError(f"Duplicate inference seed {inference_seed} for {run_dir.name}")
                inference_seeds_by_run[run_dir.name].add(inference_seed)
                bounds = report.get("query_action_offset_bounds")
                if bounds != {"min": -1, "max": 31}:
                    raise ValueError(f"Unexpected common-horizon query bounds in {report_path}: {bounds}")
                reports_by_run[run_dir.name].append(report)
    for run_name, reports in sorted(reports_by_run.items()):
        episode_ids, metrics = aggregate_available_episode_metrics(reports)
        evaluated_steps = {row["evaluated_step"] for row in evaluation_rows if row["run_name"] == run_name}
        if len(evaluated_steps) != 1:
            raise ValueError(f"Multiple evaluated checkpoint steps within {run_name}: {evaluated_steps}")
        row = {
            **runs_by_name[run_name],
            "evaluated_step": evaluated_steps.pop(),
            "num_inference_seeds": len(reports),
            "num_episodes": len(episode_ids),
        }
        for metric, values in metrics.items():
            low, high = bootstrap_mean_interval(values, rng=np.random.default_rng(0))
            seed_means = [float(report["summary"]["episode_balanced"][metric]) for report in reports]
            row[metric] = float(values.mean())
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_inference_seed_sd"] = float(np.std(seed_means))
        run_rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "v2_evaluations.csv", evaluation_rows)
    write_csv(out_dir / "v2_run_summary.csv", run_rows)
    if skipped:
        print(f"WARNING: skipped {len(skipped)} unreadable report files (husks):")
        for path, error in skipped:
            print(f"  {path}: {error}")
    print(
        f"v2 pass collected {len(evaluation_rows)} evaluations across {len(run_rows)} runs "
        f"({sum(1 for row in evaluation_rows if row['has_v2_metrics'])} with v2 metrics) "
        f"under {out_dir}"
    )





def main() -> None:
    args = parse_args()
    if args.v2_eval_roots:
        eval_roots = [Path(part) for part in args.v2_eval_roots.split(",") if part.strip()]
        if not eval_roots:
            raise SystemExit("--v2_eval_roots provided but empty")
        out_dir = args.v2_out_dir if args.v2_out_dir is not None else eval_roots[0].parent / "results"
        collect_v2_metrics(
            eval_roots,
            out_dir,
            expected_num_episodes=args.expected_num_episodes,
            expected_samples_per_episode=args.expected_samples_per_episode,
        )
        return
    train_rows = []
    validation_rows = []
    evaluation_rows = []
    reports_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    inference_seeds_by_run: dict[str, set[int]] = defaultdict(set)
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
            (args.artifact_root / args.eval_dir_name / run_dir.name).glob("seed*/*_open_loop_metrics.json")
        ):
            validate_report_checkpoint_step(report_path, run["steps"])
            evaluation_rows.append(flatten_evaluation(report_path, run))
            report = json.loads(report_path.read_text())
            inference_seed = validate_evaluation_report(
                report_path,
                report,
                expected_num_episodes=args.expected_num_episodes,
                expected_samples_per_episode=args.expected_samples_per_episode,
            )
            if inference_seed in inference_seeds_by_run[run_dir.name]:
                raise ValueError(f"Duplicate inference seed {inference_seed} for evaluation {run_dir.name}")
            inference_seeds_by_run[run_dir.name].add(inference_seed)
            bounds = report.get("query_action_offset_bounds")
            if args.eval_dir_name == "eval_common_h32" and bounds != {"min": -1, "max": 31}:
                raise ValueError(f"Unexpected common-horizon query bounds in {report_path}: {bounds}")
            reports_by_run[run_dir.name].append(report)

    variant_rows, comparison_rows, episode_data_by_run = summarize_variants(reports_by_run, runs_by_name)
    variant_seed_rows, comparison_seed_rows = summarize_training_seed_variability(
        variant_rows, comparison_rows, episode_data_by_run
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

#!/usr/bin/env python3
"""Compile the §9.2.19 cross-query prediction-stability sweep.

Input: --stability_root tree of stability-mode re-evals (16-row representative
set over the archived report checkpoints + the fresh Q-experiment runs):
<RUN>/seed1000/*_stability_metrics.json, produced by eval_open_loop_dataset.py
--stability_eval (2026-08-25). Each JSON carries per-pair overlap disagreement
between the chunks predicted at t and t+k (same inference seed for both
members), for intervals k in {1, 5, 10}, 5 anchors/episode over all 100
validation episodes.

Outputs (in --out_dir):
  stability_h10.csv     per-run × per-interval episode-balanced means + 95% CI
  stability_h10.md      markdown table for the report
And per-run compact snapshots under --per_episode_dir (repo-tracked).

Run from the repo root with `uv run` (or any python with numpy).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

METRICS = (
    "xyz_overlap_mean_mm",
    "xyz_overlap_end_mm",
    "rotation_overlap_mean_deg",
    "rotation_overlap_end_deg",
    "gripper_overlap_mean",
)
INTERVALS = (1, 5, 10)

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 0


def bootstrap_ci(per_episode: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, per_episode.shape[0], size=(N_BOOTSTRAP, per_episode.shape[0]))
    means = per_episode[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def check_protocol(run: str, report: dict) -> None:
    if report.get("mode") != "stability":
        raise ValueError(f"{run}: not a stability-mode report")
    if list(report.get("stability_intervals", [])) != list(INTERVALS):
        raise ValueError(f"{run}: intervals {report.get('stability_intervals')} != {INTERVALS}")
    if report.get("anchors_per_episode") != 5:
        raise ValueError(f"{run}: anchors_per_episode != 5")
    if report.get("control_fps") != 30.0:
        raise ValueError(f"{run}: control_fps != 30.0")
    if len(report["samples"]) != 1500:
        raise ValueError(f"{run}: {len(report['samples'])} samples, expected 1500 (500 anchors x 3)")
    if report.get("query_action_offset_bounds") != {"min": -1, "max": 31}:
        raise ValueError(f"{run}: non-canonical query bounds")
    per_k = defaultdict(int)
    for sample in report["samples"]:
        per_k[sample["interval"]] += 1
    if dict(per_k) != {k: 500 for k in INTERVALS}:
        raise ValueError(f"{run}: per-interval sample counts {dict(per_k)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stability_root",
        default="/mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/eval_stability_h10",
    )
    ap.add_argument(
        "--out_dir",
        default="/mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/results_stability_h10",
    )
    ap.add_argument(
        "--per_episode_dir",
        default=os.path.join(os.path.dirname(__file__), "repro", "per_episode_stability"),
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.per_episode_dir, exist_ok=True)

    rows = {}
    for path in sorted(glob.glob(os.path.join(args.stability_root, "*", "seed1000", "*_stability_metrics.json"))):
        run = os.path.basename(os.path.dirname(os.path.dirname(path)))
        report = json.load(open(path))
        check_protocol(run, report)
        per_ep: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for sample in report["samples"]:
            for metric in METRICS:
                per_ep[(sample["episode_index"], sample["interval"])][metric].append(sample[metric])
        episodes = sorted({ep for ep, _ in per_ep})
        rows[run] = {
            "episodes": episodes,
            "checkpoint": report["checkpoint"],
            # per-(interval, metric): episode-balanced mean + bootstrap CI
            "cells": {},
            "per_episode": {
                interval: {
                    metric: np.array(
                        [float(np.mean(per_ep[(ep, interval)][metric])) for ep in episodes]
                    )
                    for metric in METRICS
                }
                for interval in INTERVALS
            },
        }
        for interval in INTERVALS:
            for metric in METRICS:
                values = rows[run]["per_episode"][interval][metric]
                mean = float(values.mean())
                lo, hi = bootstrap_ci(values)
                rows[run]["cells"][(interval, metric)] = (mean, lo, hi)

    csv = ["run,k,xyz_overlap_mean_mm,xyz_overlap_mean_lo,xyz_overlap_mean_hi,"
           "xyz_overlap_end_mm,rotation_overlap_mean_deg,rotation_overlap_mean_lo,"
           "rotation_overlap_mean_hi,rotation_overlap_end_deg,gripper_overlap_mean"]
    md = [
        "| Run | k | XYZ mean (mm) | XYZ end (mm) | Rot mean (deg) | Rot end (deg) |",
        "|---|---:|---|---:|---|---:|",
    ]
    for run in sorted(rows):
        for interval in INTERVALS:
            get = lambda m: rows[run]["cells"][(interval, m)]  # noqa: E731
            csv.append(
                f"{run},{interval},{get('xyz_overlap_mean_mm')[0]:.2f},"
                f"{get('xyz_overlap_mean_mm')[1]:.2f},{get('xyz_overlap_mean_mm')[2]:.2f},"
                f"{get('xyz_overlap_end_mm')[0]:.2f},{get('rotation_overlap_mean_deg')[0]:.3f},"
                f"{get('rotation_overlap_mean_deg')[1]:.3f},{get('rotation_overlap_mean_deg')[2]:.3f},"
                f"{get('rotation_overlap_end_deg')[0]:.3f},{get('gripper_overlap_mean')[0]:.4f}"
            )
            md.append(
                f"| {run} | {interval} | {get('xyz_overlap_mean_mm')[0]:.2f} "
                f"[{get('xyz_overlap_mean_mm')[1]:.2f}, {get('xyz_overlap_mean_mm')[2]:.2f}] | "
                f"{get('xyz_overlap_end_mm')[0]:.2f} | "
                f"{get('rotation_overlap_mean_deg')[0]:.2f} "
                f"[{get('rotation_overlap_mean_deg')[1]:.2f}, {get('rotation_overlap_mean_deg')[2]:.2f}] | "
                f"{get('rotation_overlap_end_deg')[0]:.2f} |"
            )

    open(os.path.join(args.out_dir, "stability_h10.csv"), "w").write("\n".join(csv) + "\n")
    open(os.path.join(args.out_dir, "stability_h10.md"), "w").write("\n".join(md) + "\n")
    print(f"compiled {len(rows)} runs x {len(INTERVALS)} intervals")

    for run, entry in rows.items():
        payload = {
            "run": run,
            "checkpoint": entry["checkpoint"],
            "protocol": {
                "mode": "stability",
                "intervals": list(INTERVALS),
                "anchors_per_episode": 5,
                "episodes": len(entry["episodes"]),
                "seed": 1000,
                "fps": 30.0,
            },
            "episode_balanced": {
                str(interval): {
                    metric: entry["cells"][(interval, metric)][0] for metric in METRICS
                }
                for interval in INTERVALS
            },
            "episode_balanced_95ci": {
                str(interval): {
                    metric: list(entry["cells"][(interval, metric)][1:])
                    for metric in METRICS
                }
                for interval in INTERVALS
            },
            "per_episode": {
                str(interval): {
                    metric: entry["per_episode"][interval][metric].tolist()
                    for metric in METRICS
                }
                for interval in INTERVALS
            },
        }
        with gzip.open(os.path.join(args.per_episode_dir, f"{run}.json.gz"), "wt") as f:
            json.dump(payload, f)
    print(f"wrote {len(rows)} per-episode repro files to {args.per_episode_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

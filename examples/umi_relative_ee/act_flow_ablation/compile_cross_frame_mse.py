#!/usr/bin/env python3
"""Compile the all-checkpoint adjacent-frame direct-MSE sweep."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path

import numpy as np

HORIZONS = (10, 30)
OPENPI_RUN_HORIZONS = {
    "pi05_lora_sroi_rot6d_seed1000_0020000steps": (10,),
    "pi05_lora_sroi_rotvec_seed1000_0020000steps": (10,),
    "pi05_openpi1m_seed1000_0100001steps": (10,),
    "pi05_lora_sroi_rot6d_h30_seed1000_0020000steps": (10, 30),
    "pi05_lora_sroi_rot6d_h30_bs4_1m_seed1000_0100000steps": (10, 30),
}
H10_ONLY_TORCH_RUN_HORIZONS = {
    "lingbot_va_axis_angle_seed1000_50000steps": (10,),
    "lingbot_va_axis_angle_seed1000_100000steps": (10,),
    "lingbot_va_axis_angle_seed1000_200000steps": (10,),
}
METRICS = (
    "action_cross_frame_mse_normalized",
    "xyz_cross_frame_mse_mm2_per_dim",
    "rotvec_cross_frame_mse_deg2_per_dim",
    "rotation_geodesic_cross_frame_mse_deg2",
    "gripper_cross_frame_mse",
)
STEP_RE = re.compile(r"_(\d+)steps$")


def checkpoint_step(run: str, checkpoint: str) -> int:
    parent = Path(checkpoint).parent.name
    if parent.isdigit():
        return int(parent)
    match = STEP_RE.search(run)
    if match is None:
        raise ValueError(f"Cannot infer training step for {run}: {checkpoint}")
    return int(match.group(1))


def check_report(run: str, report: dict, horizons: tuple[int, ...]) -> None:
    if report.get("mode") != "cross_frame_mse":
        raise ValueError(f"{run}: unexpected mode {report.get('mode')}")
    if report.get("eval_horizons") != list(horizons):
        raise ValueError(f"{run}: horizons {report.get('eval_horizons')} != {horizons}")
    if report.get("pairs_per_episode") != 5 or report.get("frame_interval") != 1:
        raise ValueError(f"{run}: non-canonical pair protocol")
    if report.get("query_action_offset_bounds") != {"min": -1, "max": 31}:
        raise ValueError(f"{run}: non-canonical query bounds")
    if report.get("control_fps") != 30.0:
        raise ValueError(f"{run}: control_fps != 30")
    for horizon in horizons:
        result = report["horizons"][str(horizon)]
        summary = result["summary"]
        if summary["num_episodes"] != 100 or summary["num_pairs"] != 500:
            raise ValueError(f"{run} h{horizon}: expected 100 episodes / 500 pairs")
        if len(result["samples"]) != 500:
            raise ValueError(f"{run} h{horizon}: sample count != 500")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_root",
        default="/mnt/data/zfei/lerobot-act-flow-ablation/eval/cross_frame_mse/tree",
    )
    parser.add_argument(
        "--manifest",
        default="/mnt/data/zfei/lerobot-act-flow-ablation/eval/cross_frame_mse/manifest.tsv",
    )
    parser.add_argument(
        "--openpi_root",
        default="/mnt/data/zfei/lerobot-act-flow-ablation/eval/cross_frame_mse/openpi_tree",
    )
    parser.add_argument(
        "--lingbot_root",
        default="/mnt/data/zfei/lerobot-act-flow-ablation/eval/lingbot_cross_frame/results",
    )
    parser.add_argument(
        "--out_dir",
        default="/mnt/data/zfei/lerobot-act-flow-ablation/eval/cross_frame_mse/results",
    )
    parser.add_argument(
        "--per_episode_dir",
        default=str(Path(__file__).parent / "repro" / "per_episode_cross_frame_mse"),
    )
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    torch_expected = {
        line.split("\t", maxsplit=1)[0]
        for line in Path(args.manifest).read_text().splitlines()[1:]
    }
    expected = torch_expected | set(OPENPI_RUN_HORIZONS) | set(H10_ONLY_TORCH_RUN_HORIZONS)
    reports = {
        path.parents[1].name: path
        for path in eval_root.glob("*/seed1000/*_cross_frame_mse_metrics.json")
    }
    reports.update(
        {
            path.parents[1].name: path
            for path in Path(args.openpi_root).glob("*/seed1000/*_cross_frame_mse_metrics.json")
        }
    )
    reports.update(
        {
            path.parents[1].name: path
            for path in Path(args.lingbot_root).glob(
                "*/seed1000/*_cross_frame_mse_metrics.json"
            )
        }
    )
    missing = sorted(expected - reports.keys())
    extra = sorted(reports.keys() - expected)
    if missing or extra:
        raise ValueError(f"Manifest mismatch: {len(missing)} missing, {len(extra)} extra")

    out_dir = Path(args.out_dir)
    per_episode_dir = Path(args.per_episode_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_episode_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    scale_reference: dict[tuple[str, int], np.ndarray] = {}
    for run in sorted(expected):
        report = json.loads(reports[run].read_text())
        horizons = OPENPI_RUN_HORIZONS.get(
            run, H10_ONLY_TORCH_RUN_HORIZONS.get(run, HORIZONS)
        )
        check_report(run, report, horizons)
        compact = {
            "run": run,
            "checkpoint": report["checkpoint"],
            "policy_type": report["policy_type"],
            "protocol": {
                "mode": report["mode"],
                "horizons": report["eval_horizons"],
                "pairs_per_episode": report["pairs_per_episode"],
                "frame_interval": report["frame_interval"],
                "seed": report["seed"],
            },
            "horizons": {},
        }
        source = "openpi" if run in OPENPI_RUN_HORIZONS else "torch"
        for horizon in horizons:
            result = report["horizons"][str(horizon)]
            scales = np.asarray(result["normalization"]["per_dim_half_ranges"])
            scale_key = (source, horizon)
            if scale_key in scale_reference and not np.allclose(
                scales, scale_reference[scale_key], rtol=1e-9, atol=1e-12
            ):
                raise ValueError(f"{run} h{horizon}: GT normalization scales differ")
            scale_reference.setdefault(scale_key, scales)
            means = result["summary"]["episode_balanced"]
            intervals = result["summary"]["episode_balanced_95ci"]
            row = {
                "run": run,
                "policy_type": report["policy_type"],
                "step": checkpoint_step(run, report["checkpoint"]),
                "horizon": horizon,
                "checkpoint": report["checkpoint"],
            }
            for metric in METRICS:
                row[metric] = means[metric]
                row[f"{metric}_lo"] = intervals[metric]["low"]
                row[f"{metric}_hi"] = intervals[metric]["high"]
            rows.append(row)
            compact["horizons"][str(horizon)] = {
                "normalization": result["normalization"],
                "episode_balanced": means,
                "episode_balanced_95ci": intervals,
                "per_episode": result["summary"]["per_episode"],
            }
        with gzip.open(per_episode_dir / f"{run}.json.gz", "wt") as handle:
            json.dump(compact, handle)

    columns = ["run", "policy_type", "step", "horizon", "checkpoint"]
    for metric in METRICS:
        columns.extend((metric, f"{metric}_lo", f"{metric}_hi"))
    csv_path = out_dir / "cross_frame_mse.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "| Run | h | Action MSE (norm.) | XYZ MSE (mm²/dim) | Rot. geodesic MSE (deg²) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['run']} | {row['horizon']} | "
            f"{row['action_cross_frame_mse_normalized']:.3f} | "
            f"{row['xyz_cross_frame_mse_mm2_per_dim']:.3f} | "
            f"{row['rotation_geodesic_cross_frame_mse_deg2']:.3f} |"
        )
    (out_dir / "cross_frame_mse.md").write_text("\n".join(markdown) + "\n")
    print(f"compiled {len(expected)} checkpoints / {len(rows)} native horizon rows to {csv_path}")


if __name__ == "__main__":
    main()

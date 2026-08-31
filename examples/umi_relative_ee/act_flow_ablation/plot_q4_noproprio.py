#!/usr/bin/env python3
"""§9.2.18 dedicated figure — Q4 no-proprio ACT R50-V1 vs the matched 1-frame curve.

Three matched-budget figures over the five shared checkpoints (100k..500k):
  1. Six canonical-h10 co-primary accuracy metrics.
  2. Six canonical-h10 physical motion metrics (velocity/acceleration/jerk).
  3. Adjacent-frame direct prediction-change MSE at h10 and h30.

All panels include 95% episode-bootstrap CI bands where available.

Data: unified_h10_run_summary.csv (recompiled 2026-08-26 with the five Q4
rows). Run via:
  /home/zfei/anaconda3/envs/py312/bin/python \\
    examples/umi_relative_ee/act_flow_ablation/plot_q4_noproprio.py
"""
from __future__ import annotations

import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
UNIFIED_CSV = "/mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/results/unified_h10_run_summary.csv"
PHYSICAL_CSV = os.path.join(HERE, "repro", "physical_dynamics_h10.csv")
CROSS_FRAME_CSV = os.path.join(HERE, "repro", "cross_frame_mse.csv")
FIG_DIR = os.path.join(HERE, "figures")

C_1F = "#1f77b4"
C_NP = "#8c564b"

PANELS = [
    ("xyz_end_m", "XYZ endpoint error (mm)", 1000),
    ("rotation_end_deg", "Rotation endpoint error (deg)", 1),
    ("xyz_l1_per_dim_m", "XYZ L1 per dimension (mm)", 1000),
    ("rotvec_l1_per_dim_deg", "Rotation L1 per dimension (deg)", 1),
    ("action_acc_at_0p5", "Action Acc@0.5", 1),
    ("action_acc_at_0p1", "Action Acc@0.1", 1),
]

PHYSICAL_PANELS = [
    ("rot_vel_deg_s", "Rotational velocity (deg/s)"),
    ("rot_accel_deg_s2", "Rotational acceleration (deg/s²)"),
    ("rot_jerk_deg_s3", "Rotational jerk (deg/s³)"),
    ("xyz_vel_mm_s", "XYZ velocity (mm/s)"),
    ("xyz_accel_mm_s2", "XYZ acceleration (mm/s²)"),
    ("xyz_jerk_mm_s3", "XYZ jerk (mm/s³)"),
]

CROSS_FRAME_PANELS = [
    ("action_cross_frame_mse_normalized", "Normalized action MSE"),
    ("xyz_cross_frame_mse_mm2_per_dim", "XYZ MSE (mm² per dim)"),
    ("rotation_geodesic_cross_frame_mse_deg2", "Rotation geodesic MSE (deg²)"),
]


def load(path: str, prefix: str) -> list[tuple[int, dict]]:
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["run"].startswith(prefix)]
    key = "step" if "step" in rows[0] else "steps"
    return sorted((int(r[key]), r) for r in rows)


def curve(ax, series, metric, scale, color, label, marker, ls="-"):
    xs = [s for s, _ in series]
    ys = [float(r[metric]) * scale for _, r in series]
    ax.plot(xs, ys, marker=marker, ms=5, lw=1.8, ls=ls, color=color, label=label)
    if f"{metric}__ci_low" in series[0][1]:
        ax.fill_between(
            xs,
            [float(r[f"{metric}__ci_low"]) * scale for _, r in series],
            [float(r[f"{metric}__ci_high"]) * scale for _, r in series],
            color=color, alpha=0.15, lw=0,
        )
    elif f"{metric}_lo" in series[0][1]:
        ax.fill_between(
            xs,
            [float(r[f"{metric}_lo"]) * scale for _, r in series],
            [float(r[f"{metric}_hi"]) * scale for _, r in series],
            color=color, alpha=0.15, lw=0,
        )


def style_budget_axes(axes, ticks: list[int]) -> None:
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{step // 1000}k" for step in ticks])
        ax.minorticks_off()
        ax.set_xlabel("training steps")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)


def matched_series(path: str) -> tuple[list[tuple[int, dict]], list[tuple[int, dict]]]:
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    step_key = "step" if "step" in rows[0] else "steps"
    baseline = sorted(
        (int(row[step_key]), row)
        for row in rows
        if re.fullmatch(r"act_r50_v1_vae_seed1000_\d{7}steps", row["run"])
    )
    no_proprio = sorted(
        (int(row[step_key]), row)
        for row in rows
        if re.fullmatch(r"act_r50_v1_vae_noproprio_seed1000_\d{7}steps", row["run"])
    )
    shared_steps = {step for step, _ in no_proprio}
    baseline = [(step, row) for step, row in baseline if step in shared_steps]
    if [step for step, _ in baseline] != [step for step, _ in no_proprio]:
        raise ValueError(f"Q4 and baseline rows in {path} do not have the same budgets")
    return baseline, no_proprio


def fig_accuracy(one_u: list[tuple[int, dict]], np_u: list[tuple[int, dict]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Q4: does ACT need proprioception? — R50-V1 image-only vs matched proprioceptive ACT, canonical h10",
        fontsize=12,
    )
    for ax, (metric, title, scale) in zip(axes.flat, PANELS, strict=True):
        curve(ax, one_u, metric, scale, C_1F, "1-frame + state", "s")
        curve(ax, np_u, metric, scale, C_NP, "image-only (no proprioception)", "P")
        ax.set_title(title, fontsize=10)
    style_budget_axes(axes, [step for step, _ in np_u])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG_DIR, "q4_noproprio_vs_1frame.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def fig_physical() -> None:
    one_p, np_p = matched_series(PHYSICAL_CSV)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Q4 physical motion dynamics — image-only vs matched proprioceptive ACT, canonical h10 (dt = 1/30 s)",
        fontsize=12,
    )
    for ax, (metric, title) in zip(axes.flat, PHYSICAL_PANELS, strict=True):
        curve(ax, one_p, metric, 1, C_1F, "1-frame + state", "s")
        curve(ax, np_p, metric, 1, C_NP, "image-only (no proprioception)", "P")
        gt = float(one_p[0][1][f"gt_{metric}"])
        ax.axhline(gt, color="k", ls="--", lw=1.2, label=f"demonstrated ({gt:,.1f})")
        ax.set_title(title, fontsize=10)
    style_budget_axes(axes, [step for step, _ in np_p])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG_DIR, "q4_noproprio_physical_dynamics.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def fig_cross_frame() -> None:
    with open(CROSS_FRAME_CSV) as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex="col")
    fig.suptitle(
        "Q4 adjacent-frame prediction consistency — direct change from query t to t+1; lower is more stable",
        fontsize=12,
    )
    ticks: list[int] | None = None
    for row_index, horizon in enumerate((10, 30)):
        horizon_rows = [row for row in rows if int(row["horizon"]) == horizon]
        one_c = sorted(
            (int(row["step"]), row)
            for row in horizon_rows
            if re.fullmatch(r"act_r50_v1_vae_seed1000_\d{7}steps", row["run"])
        )
        np_c = sorted(
            (int(row["step"]), row)
            for row in horizon_rows
            if re.fullmatch(r"act_r50_v1_vae_noproprio_seed1000_\d{7}steps", row["run"])
        )
        shared_steps = {step for step, _ in np_c}
        one_c = [(step, row) for step, row in one_c if step in shared_steps]
        if [step for step, _ in one_c] != [step for step, _ in np_c]:
            raise ValueError(f"Q4 and baseline cross-frame h{horizon} budgets differ")
        ticks = [step for step, _ in np_c]
        for column_index, (metric, title) in enumerate(CROSS_FRAME_PANELS):
            ax = axes[row_index, column_index]
            curve(ax, one_c, metric, 1, C_1F, "1-frame + state", "s")
            curve(ax, np_c, metric, 1, C_NP, "image-only (no proprioception)", "P")
            ax.set_yscale("log")
            ax.set_title(f"h{horizon}: {title}", fontsize=10)
    if ticks is None:
        raise ValueError("No Q4 cross-frame rows found")
    style_budget_axes(axes, ticks)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG_DIR, "q4_noproprio_cross_frame_mse.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    one_u = load(UNIFIED_CSV, "act_r50_v1_vae_seed1000_")
    np_u = load(UNIFIED_CSV, "act_r50_v1_vae_noproprio_seed1000_")
    shared_steps = {step for step, _ in np_u}
    one_u = [(step, row) for step, row in one_u if step in shared_steps]
    if [step for step, _ in one_u] != [step for step, _ in np_u]:
        raise ValueError("Q4 and baseline series do not contain the same training budgets")

    fig_accuracy(one_u, np_u)
    fig_physical()
    fig_cross_frame()
    print(f"Q4 suite uses {len(np_u)} no-proprio vs {len(one_u)} one-frame points")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""§9.2.16 LingBot-VA budget figures in the report-standard comparison format.

Renders three views against the two budget-covered anchors, ACT R50-VAE
(ImageNet-V1) and the π0.5 port:
  1. Six canonical-h10 co-primary accuracy metrics.
  2. Six canonical-h10 physical motion metrics (velocity/acceleration/jerk).
  3. Three adjacent-frame direct prediction-change metrics at h10.

LingBot-VA emits a 16-action chunk, so h30 evaluation is structurally
unsupported. All panels include 95% episode-bootstrap CI bands.
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

C_LING = "#ff9896"
C_ACT = "#1f77b4"
C_PORT = "#ff7f0e"

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


def load(path: str, pattern: str, horizon: int | None = None) -> list[tuple[int, dict]]:
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    key = "step" if "step" in rows[0] else "steps"
    return sorted(
        (int(row[key]), row)
        for row in rows
        if re.fullmatch(pattern, row["run"])
        and (horizon is None or int(row["horizon"]) == horizon)
    )


def curve(ax, series, metric, scale, color, label, marker):
    if not series:
        raise ValueError(f"No rows available for {label}: {metric}")
    xs = [step for step, _ in series]
    ys = [float(row[metric]) * scale for _, row in series]
    ax.plot(xs, ys, marker=marker, ms=5, lw=1.8, color=color, label=label)
    if f"{metric}__ci_low" in series[0][1]:
        low_key, high_key = f"{metric}__ci_low", f"{metric}__ci_high"
    elif f"{metric}_lo" in series[0][1]:
        low_key, high_key = f"{metric}_lo", f"{metric}_hi"
    else:
        return
    ax.fill_between(
        xs,
        [float(row[low_key]) * scale for _, row in series],
        [float(row[high_key]) * scale for _, row in series],
        color=color,
        alpha=0.15,
        lw=0,
    )


def style_axes(axes) -> None:
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlabel("training steps")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)


def add_curves(ax, act, port, ling, metric, scale=1) -> None:
    curve(ax, act, metric, scale, C_ACT, "ACT R50-VAE 1-frame", "s")
    curve(ax, port, metric, scale, C_PORT, "π0.5 port", "^")
    curve(ax, ling, metric, scale, C_LING, "LingBot-VA", "*")


def fig_accuracy() -> int:
    ling = load(UNIFIED_CSV, r"lingbot_va_axis_angle_seed1000_\d+steps")
    act = load(UNIFIED_CSV, r"act_r50_v1_vae_seed1000_\d{7}steps")
    port = load(UNIFIED_CSV, r"pi05_port_seed1000_\d{7}steps")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "LingBot-VA vs budget-covered anchors — six canonical co-primary metrics at h10",
        fontsize=12,
    )
    for ax, (metric, title, scale) in zip(axes.flat, PANELS, strict=True):
        add_curves(ax, act, port, ling, metric, scale)
        ax.set_title(title, fontsize=10)
    style_axes(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG_DIR, "lingbot_budget.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")
    return len(ling)


def fig_physical() -> None:
    ling = load(PHYSICAL_CSV, r"lingbot_va_axis_angle_seed1000_\d+steps")
    act = load(PHYSICAL_CSV, r"act_r50_v1_vae_seed1000_\d{7}steps")
    port = load(PHYSICAL_CSV, r"pi05_port_seed1000_\d{7}steps")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "LingBot-VA physical motion dynamics vs budget-covered anchors, canonical h10 (dt = 1/30 s)",
        fontsize=12,
    )
    for ax, (metric, title) in zip(axes.flat, PHYSICAL_PANELS, strict=True):
        add_curves(ax, act, port, ling, metric)
        gt = float(act[0][1][f"gt_{metric}"])
        ax.axhline(gt, color="k", ls="--", lw=1.2, label=f"demonstrated ({gt:,.1f})")
        ax.set_title(title, fontsize=10)
    style_axes(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG_DIR, "lingbot_physical_dynamics.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def fig_cross_frame() -> None:
    ling = load(CROSS_FRAME_CSV, r"lingbot_va_axis_angle_seed1000_\d+steps", horizon=10)
    act = load(CROSS_FRAME_CSV, r"act_r50_v1_vae_seed1000_\d{7}steps", horizon=10)
    port = load(CROSS_FRAME_CSV, r"pi05_port_seed1000_\d{7}steps", horizon=10)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle(
        "LingBot-VA adjacent-frame prediction consistency at h10 — direct change from query t to t+1; lower is more stable",
        fontsize=12,
    )
    for ax, (metric, title) in zip(axes, CROSS_FRAME_PANELS, strict=True):
        add_curves(ax, act, port, ling, metric)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
    style_axes(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(FIG_DIR, "lingbot_cross_frame_mse.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    lingbot_points = fig_accuracy()
    fig_physical()
    fig_cross_frame()
    print(f"Fig. 35 suite uses {lingbot_points} LingBot-VA checkpoints")


if __name__ == "__main__":
    main()

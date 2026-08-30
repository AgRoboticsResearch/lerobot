#!/usr/bin/env python3
"""§9.2.20 dedicated figure — Q4 state-window sweep W ∈ {0, 2, 3, 4, 5, 10}.

Two matched-budget figures over the five shared checkpoints (100k..500k):
  1. Six canonical-h10 co-primary accuracy metrics.
  2. Six canonical-h10 physical motion metrics (velocity/acceleration/jerk).

W counts total poses in the UMI proprio state incl. the identity-current pose
(state dim 10*W): W=0 is the image-only no-proprio arm, W=2 the historical
two-pose baseline, W=3/4/5/10 the window extensions. All panels include 95% episode-bootstrap CI bands.

The adjacent-frame cross-frame panel is deferred: the §9.2.19 sweep has not
been extended to the W=5/W=10 arms.

Data: unified_h10_run_summary.csv (recompiled 2026-08-29 with the ten
W=5/W=10 rows and again with the five W=3 rows). Run via:
  /home/zfei/anaconda3/envs/py312/bin/python \\
    examples/umi_relative_ee/act_flow_ablation/plot_q4_state_window.py
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
FIG_DIR = os.path.join(HERE, "figures")

# (prefix, label, color, marker) — W ascending.
SERIES = [
    (r"act_r50_v1_vae_noproprio_seed1000_\d{7}steps", "W=0 (no proprio)", "#8c564b", "P"),
    (r"act_r50_v1_vae_seed1000_\d{7}steps", "W=2 (baseline)", "#1f77b4", "s"),
    (r"act_r50_v1_vae_state3_seed1000_\d{7}steps", "W=3", "#bcbd22", "+"),
    (r"act_r50_v1_vae_state4_seed1000_\d{7}steps", "W=4", "#17becf", "*"),
    (r"act_r50_v1_vae_state5_seed1000_\d{7}steps", "W=5", "#e6550d", "<"),
    (r"act_r50_v1_vae_state10_seed1000_\d{7}steps", "W=10", "#756bb1", ">"),
]

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


def matched_series(path: str) -> list[tuple[str, list[tuple[int, dict]]]]:
    """Load every W arm restricted to the budgets shared by ALL arms."""
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    step_key = "step" if "step" in rows[0] else "steps"
    arms = []
    for pattern, _, _, _ in SERIES:
        arm = sorted(
            (int(row[step_key]), row) for row in rows if re.fullmatch(pattern, row["run"])
        )
        if not arm:
            raise ValueError(f"No rows match {pattern} in {path}")
        arms.append(arm)
    shared = set.intersection(*( {step for step, _ in arm} for arm in arms ))
    matched = [[(step, row) for step, row in arm if step in shared] for arm in arms]
    lengths = {len(arm) for arm in matched}
    if len(lengths) != 1:
        raise ValueError(f"State-window arms in {path} do not share budgets: {lengths}")
    return [(label, arm) for (_, label, _, _), arm in zip(SERIES, matched, strict=True)]


def curve(ax, series, metric, scale, color, label, marker):
    xs = [s for s, _ in series]
    ys = [float(r[metric]) * scale for _, r in series]
    ax.plot(xs, ys, marker=marker, ms=5, lw=1.8, color=color, label=label)
    if f"{metric}__ci_low" in series[0][1]:
        lo_key, hi_key = f"{metric}__ci_low", f"{metric}__ci_high"
    elif f"{metric}_lo" in series[0][1]:
        lo_key, hi_key = f"{metric}_lo", f"{metric}_hi"
    else:
        return
    ax.fill_between(
        xs,
        [float(r[lo_key]) * scale for _, r in series],
        [float(r[hi_key]) * scale for _, r in series],
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


def fig_accuracy(arms) -> int:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Q4 state-window sweep — proprio history length W, ACT R50-V1, canonical h10",
        fontsize=12,
    )
    for ax, (metric, title, scale) in zip(axes.flat, PANELS, strict=True):
        for (label, arm), (_, _, color, marker) in zip(arms, SERIES, strict=True):
            curve(ax, arm, metric, scale, color, label, marker)
        ax.set_title(title, fontsize=10)
    style_budget_axes(axes, [step for step, _ in arms[-1][1]])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG_DIR, "q4_state_window_accuracy.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")
    return len(arms[-1][1])


def fig_physical(arms) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Q4 state-window physical dynamics — W ∈ {0,2,3,4,5,10}, canonical h10 (dt = 1/30 s)",
        fontsize=12,
    )
    for ax, (metric, title) in zip(axes.flat, PHYSICAL_PANELS, strict=True):
        for (label, arm), (_, _, color, marker) in zip(arms, SERIES, strict=True):
            curve(ax, arm, metric, 1, color, label, marker)
        gt = float(arms[1][1][0][1][f"gt_{metric}"])
        ax.axhline(gt, color="k", ls="--", lw=1.2, label=f"demonstrated ({gt:,.1f})")
        ax.set_title(title, fontsize=10)
    style_budget_axes(axes, [step for step, _ in arms[-1][1]])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG_DIR, "q4_state_window_physical_dynamics.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    unified_arms = matched_series(UNIFIED_CSV)
    physical_arms = matched_series(PHYSICAL_CSV)
    points = fig_accuracy(unified_arms)
    fig_physical(physical_arms)
    print(f"Q4 state-window suite uses {len(SERIES)} arms x {points} shared checkpoints")


if __name__ == "__main__":
    main()

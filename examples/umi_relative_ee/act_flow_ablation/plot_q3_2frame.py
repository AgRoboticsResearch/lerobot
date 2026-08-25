#!/usr/bin/env python3
"""§9.2.17 dedicated figure — Q3 two-frame ACT R50-V1 vs the matched 1-frame curve.

Three budget panels over the five shared checkpoints (100k..500k):
  1. XYZ endpoint error (canonical h10, 95% bootstrap CI bands)
  2. Within-chunk rotational 2nd difference (v2 jerk)
  3. Physical rotational jerk (deg/s^3, dt = 1/30 s)

Data: unified_h10_run_summary.csv + physical_jerk_h10.csv (both recompiled
2026-08-25 with the five Q3 rows). Run via:
  /home/zfei/anaconda3/envs/py312/bin/python \\
    examples/umi_relative_ee/act_flow_ablation/plot_q3_2frame.py
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
UNIFIED_CSV = "/mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/results/unified_h10_run_summary.csv"
PHYS_CSV = "/mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/results_physical_jerk/physical_jerk_h10.csv"
FIG_DIR = os.path.join(HERE, "figures")

C_1F = "#1f77b4"
C_2F = "#1f9e89"


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


def main() -> None:
    one_u = load(UNIFIED_CSV, "act_r50_v1_vae_seed1000_")
    two_u = load(UNIFIED_CSV, "act_r50_v1_vae_2frame_seed1000_")
    one_p = load(PHYS_CSV, "act_r50_v1_vae_seed1000_")
    two_p = load(PHYS_CSV, "act_r50_v1_vae_2frame_seed1000_")
    gt_jerk = float(one_p[0][1]["gt_rot_jerk_deg_s3"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(
        "Q3: does a 2-frame input help? — ACT R50-V1 2-frame (t−1, t) vs matched 1-frame, canonical h10",
        fontsize=12,
    )
    curve(axes[0], one_u, "xyz_end_m", 1000, C_1F, "1-frame (report curve)", "s")
    curve(axes[0], two_u, "xyz_end_m", 1000, C_2F, "2-frame (Q3)", "x")
    axes[0].set_title("XYZ endpoint error (mm)", fontsize=10)
    curve(axes[1], one_u, "rot_jerk_deg", 1, C_1F, "1-frame", "s")
    curve(axes[1], two_u, "rot_jerk_deg", 1, C_2F, "2-frame", "x")
    axes[1].axhline(float(one_u[0][1]["gt_rot_jerk_deg"]), color="k", ls="--", lw=1.2, label="demonstrated")
    axes[1].set_title("Within-chunk rotational 2nd-diff (deg/step²)", fontsize=10)
    curve(axes[2], one_p, "rot_jerk_deg_s3", 1, C_1F, "1-frame", "s")
    curve(axes[2], two_p, "rot_jerk_deg_s3", 1, C_2F, "2-frame", "x")
    axes[2].axhline(gt_jerk, color="k", ls="--", lw=1.2, label=f"demonstrated ({gt_jerk:,.0f})")
    axes[2].set_title("Physical rotational jerk (deg/s³)", fontsize=10)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([s for s, _ in two_u])
        ax.set_xticklabels([f"{s // 1000}k" for s, _ in two_u])
        ax.minorticks_off()
        ax.set_xlabel("training steps")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(FIG_DIR, "q3_2frame_vs_1frame.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out} ({len(two_u)} two-frame vs {len(one_u)} one-frame points)")


if __name__ == "__main__":
    main()

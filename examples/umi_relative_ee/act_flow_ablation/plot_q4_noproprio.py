#!/usr/bin/env python3
"""§9.2.18 dedicated figure — Q4 no-proprio ACT R50-V1 vs the matched 1-frame curve.

Four budget panels over the five shared checkpoints (100k..500k):
  1. XYZ endpoint error (canonical h10, 95% bootstrap CI bands)
  2. Action Acc@0.1
  3. Within-chunk rotational 2nd difference (v2 jerk)
  4. Physical rotational jerk (deg/s^3, dt = 1/30 s)

Data: unified_h10_run_summary.csv + physical_jerk_h10.csv (both recompiled
2026-08-26 with the five Q4 rows). Run via:
  /home/zfei/anaconda3/envs/py312/bin/python \\
    examples/umi_relative_ee/act_flow_ablation/plot_q4_noproprio.py
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
C_NP = "#8c564b"


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
    np_u = load(UNIFIED_CSV, "act_r50_v1_vae_noproprio_seed1000_")
    one_p = load(PHYS_CSV, "act_r50_v1_vae_seed1000_")
    np_p = load(PHYS_CSV, "act_r50_v1_vae_noproprio_seed1000_")
    gt_jerk = float(one_p[0][1]["gt_rot_jerk_deg_s3"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        "Q4: does ACT need proprioception? — R50-V1 image-only (no-proprio) vs matched 1-frame, canonical h10",
        fontsize=12,
    )
    curve(axes[0][0], one_u, "xyz_end_m", 1000, C_1F, "1-frame + state (report curve)", "s")
    curve(axes[0][0], np_u, "xyz_end_m", 1000, C_NP, "no-proprio (Q4)", "P")
    axes[0][0].set_title("XYZ endpoint error (mm)", fontsize=10)
    curve(axes[0][1], one_u, "action_acc_at_0p1", 1, C_1F, "1-frame + state", "s")
    curve(axes[0][1], np_u, "action_acc_at_0p1", 1, C_NP, "no-proprio (Q4)", "P")
    axes[0][1].set_title("Action Acc@0.1", fontsize=10)
    curve(axes[1][0], one_u, "rot_jerk_deg", 1, C_1F, "1-frame + state", "s")
    curve(axes[1][0], np_u, "rot_jerk_deg", 1, C_NP, "no-proprio (Q4)", "P")
    axes[1][0].axhline(float(one_u[0][1]["gt_rot_jerk_deg"]), color="k", ls="--", lw=1.2, label="demonstrated")
    axes[1][0].set_title("Within-chunk rotational 2nd-diff (deg/step²)", fontsize=10)
    curve(axes[1][1], one_p, "rot_jerk_deg_s3", 1, C_1F, "1-frame + state", "s")
    curve(axes[1][1], np_p, "rot_jerk_deg_s3", 1, C_NP, "no-proprio (Q4)", "P")
    axes[1][1].axhline(gt_jerk, color="k", ls="--", lw=1.2, label=f"demonstrated ({gt_jerk:,.0f})")
    axes[1][1].set_title("Physical rotational jerk (deg/s³)", fontsize=10)
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xticks([s for s, _ in np_u])
        ax.set_xticklabels([f"{s // 1000}k" for s, _ in np_u])
        ax.minorticks_off()
        ax.set_xlabel("training steps")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG_DIR, "q4_noproprio_vs_1frame.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out} ({len(np_u)} no-proprio vs {len(one_u)} one-frame points)")


if __name__ == "__main__":
    main()

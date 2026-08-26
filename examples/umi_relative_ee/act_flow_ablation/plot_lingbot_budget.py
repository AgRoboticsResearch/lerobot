#!/usr/bin/env python3
"""§9.2.16 dedicated figure — LingBot-VA 3-point budget curve vs reference families.

Four panels over the three scored budgets (50k/100k/200k):
  1. XYZ endpoint error (canonical h10, 95% bootstrap CI bands)
  2. Action Acc@0.1
  3. Within-chunk rotational 2nd difference (v2 jerk)
  4. Physical rotational jerk (deg/s^3, dt = 1/30 s)

References: the ACT R50-VAE (ImageNet-V1) 1-frame curve and the π0.5 port
curve — the report's two budget-covered anchors.

Data: unified_h10_run_summary.csv + physical_jerk_h10.csv (recompiled
2026-08-26 with the three LingBot rows). Run via:
  /home/zfei/anaconda3/envs/py312/bin/python \\
    examples/umi_relative_ee/act_flow_ablation/plot_lingbot_budget.py
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

C_LING = "#ff9896"
C_ACT = "#1f77b4"
C_PORT = "#ff7f0e"


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
    ling_u = load(UNIFIED_CSV, "lingbot_va_axis_angle_seed1000_")
    act_u = load(UNIFIED_CSV, "act_r50_v1_vae_seed1000_")
    port_u = load(UNIFIED_CSV, "pi05_port_seed1000_")
    ling_p = load(PHYS_CSV, "lingbot_va_axis_angle_seed1000_")
    act_p = load(PHYS_CSV, "act_r50_v1_vae_seed1000_")
    port_p = load(PHYS_CSV, "pi05_port_seed1000_")
    gt_jerk = float(act_p[0][1]["gt_rot_jerk_deg_s3"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        "LingBot-VA (video-action, 16-frame chunk) vs reference budget curves — canonical h10",
        fontsize=12,
    )
    curve(axes[0][0], act_u, "xyz_end_m", 1000, C_ACT, "ACT R50-VAE 1-frame", "s")
    curve(axes[0][0], port_u, "xyz_end_m", 1000, C_PORT, "π0.5 port", "^")
    curve(axes[0][0], ling_u, "xyz_end_m", 1000, C_LING, "LingBot-VA", "*")
    axes[0][0].set_title("XYZ endpoint error (mm)", fontsize=10)
    curve(axes[0][1], act_u, "action_acc_at_0p1", 1, C_ACT, "ACT R50-VAE 1-frame", "s")
    curve(axes[0][1], port_u, "action_acc_at_0p1", 1, C_PORT, "π0.5 port", "^")
    curve(axes[0][1], ling_u, "action_acc_at_0p1", 1, C_LING, "LingBot-VA", "*")
    axes[0][1].set_title("Action Acc@0.1", fontsize=10)
    curve(axes[1][0], act_u, "rot_jerk_deg", 1, C_ACT, "ACT R50-VAE 1-frame", "s")
    curve(axes[1][0], port_u, "rot_jerk_deg", 1, C_PORT, "π0.5 port", "^")
    curve(axes[1][0], ling_u, "rot_jerk_deg", 1, C_LING, "LingBot-VA", "*")
    axes[1][0].axhline(float(act_u[0][1]["gt_rot_jerk_deg"]), color="k", ls="--", lw=1.2, label="demonstrated")
    axes[1][0].set_title("Within-chunk rotational 2nd-diff (deg/step²)", fontsize=10)
    curve(axes[1][1], act_p, "rot_jerk_deg_s3", 1, C_ACT, "ACT R50-VAE 1-frame", "s")
    curve(axes[1][1], port_p, "rot_jerk_deg_s3", 1, C_PORT, "π0.5 port", "^")
    curve(axes[1][1], ling_p, "rot_jerk_deg_s3", 1, C_LING, "LingBot-VA", "*")
    axes[1][1].axhline(gt_jerk, color="k", ls="--", lw=1.2, label=f"demonstrated ({gt_jerk:,.0f})")
    axes[1][1].set_title("Physical rotational jerk (deg/s³)", fontsize=10)
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlabel("training steps")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG_DIR, "lingbot_budget.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out} ({len(ling_u)} lingbot vs {len(act_u)} ACT / {len(port_u)} port points)")


if __name__ == "__main__":
    main()

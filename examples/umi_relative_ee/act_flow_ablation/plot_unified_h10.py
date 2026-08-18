#!/usr/bin/env python3
"""Figures for the unified horizon-10 re-evaluation (§9.2.9).

Reads the compiler's unified_h10_run_summary.csv (produced by
compile_unified_h10.py after all protocol assertions passed) and renders:

  figures/unified_h10_metrics.png — one representative bar per surviving
    model family across the six co-primary metrics, with 95% episode
    bootstrap CIs (openpi rows use the evaluator's own CIs).
  figures/unified_h10_budget.png  — t+10 budget curves: historical R18-VAE
    (30 checkpoints), fresh R50-V1 (100k-spaced), and single-budget
    references (openpi arms, port-recipe Arm B, seed-23k companions) on
    acc@0.1 and XYZ endpoint.

Rows absent from the CSV (R50-V1 900k/1M before the trainer finishes; kiwi
port/SmolVLA rows before K4) are skipped gracefully and appear on the next
render. Run via:
  MPLCONFIGDIR=/tmp/lerobot-matplotlib uv run --with matplotlib python \
    examples/umi_relative_ee/act_flow_ablation/plot_unified_h10.py
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = "/mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/results/unified_h10_run_summary.csv"
FIG_DIR = os.path.join(HERE, "figures")

# family -> color (stable across renders)
COLORS = {
    "hist": "#444444",
    "r50v1": "#1f77b4",
    "actl1": "#2ca02c",
    "r50vae": "#9467bd",
    "flow": "#d62728",
    "port": "#ff7f0e",
    "openpi": "#17becf",
    "smol": "#8c564b",
}

# Bar-chart representatives: (run, label, family). One row per surviving
# family-recipe at its final/available budget; companions keep both training
# seeds to display seed spread.
REPRESENTATIVES = [
    ("act_umi_identity_rot6d_1459_3000000steps", "ACT R18-VAE 3M (historical)", "hist"),
    ("act_r50_v1_vae_seed1000_0800000steps", "ACT R50-V1 800k", "r50v1"),
    ("act_r18_l1_seed2000_100000steps", "ACT-L1 100k s2000", "actl1"),
    ("act_r18_l1_seed3000_100000steps", "ACT-L1 100k s3000", "actl1"),
    ("act_r50_vae_seed2000_100000steps", "ACT R50-VAE 80k s2000", "r50vae"),
    ("act_r50_vae_seed3000_100000steps", "ACT R50-VAE 80k s3000", "r50vae"),
    ("act_r18_flow_u_lr1e5_seed2000_100000steps", "ACT-flow 50k s2000", "flow"),
    ("act_r18_flow_u_lr1e5_seed3000_100000steps", "ACT-flow 50k s3000", "flow"),
    ("pi05_port_openpi_recipe_seed1000_020000steps", "π0.5 port o-recipe 20k", "port"),
    ("pi05_lora_sroi_rot6d_seed1000_0020000steps", "openpi rot6d 20k", "openpi"),
    ("pi05_lora_sroi_rotvec_seed1000_0020000steps", "openpi rotvec 20k", "openpi"),
    ("pi05_lora_sroi_rot6d_h30_seed1000_0020000steps", "openpi rot6d-h30 20k", "openpi"),
    # kiwi rows (K4): rendered when present
    ("pi05_port_seed1000_1000000steps", "π0.5 port 1M (kiwi)", "port"),
    ("pi05_port_seed1000_0700000steps", "π0.5 port 700K (kiwi)", "port"),
    ("smolvla_rot6d_seed1000_100000steps", "SmolVLA rot6d 100k", "smol"),
    ("smolvla_axis_angle_seed1000_100000steps", "SmolVLA axis-angle 100k", "smol"),
]

PANELS = [
    ("xyz_end_m", "XYZ endpoint error (mm)", 1000, "{:.1f}"),
    ("rotation_end_deg", "Rotation endpoint error (deg)", 1, "{:.2f}"),
    ("xyz_l1_per_dim_m", "XYZ L1 per dim (mm)", 1000, "{:.2f}"),
    ("rotvec_l1_per_dim_deg", "Rotvec L1 per dim (deg)", 1, "{:.2f}"),
    ("action_acc_at_0p5", "accuracy@0.5 (action)", 1, "{:.3f}"),
    ("action_acc_at_0p1", "accuracy@0.1 (action)", 1, "{:.3f}"),
]


def load_rows() -> dict[str, dict]:
    with open(CSV_PATH) as f:
        return {r["run"]: r for r in csv.DictReader(f)}


def fig_metrics(rows: dict[str, dict]) -> None:
    reps = [(run, label, fam) for run, label, fam in REPRESENTATIVES if run in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle(
        "Unified horizon-10 re-evaluation — all surviving models, one protocol "
        "(500 queries, endpoint = t+10, 95% episode bootstrap CI)",
        fontsize=12,
    )
    for ax, (met, title, scale, _) in zip(axes.flat, PANELS):
        labels, means, lows, highs, colors = [], [], [], [], []
        for run, label, fam in reps:
            r = rows[run]
            v = float(r[met]) * scale
            labels.append(label)
            means.append(v)
            lo = r.get(f"{met}__ci_low")
            hi = r.get(f"{met}__ci_high")
            lows.append(v - float(lo) * scale if lo else 0.0)
            highs.append(float(hi) * scale - v if hi else 0.0)
            colors.append(COLORS[fam])
        y = range(len(labels))
        ax.barh(y, means, xerr=[lows, highs], color=colors, alpha=0.85,
                error_kw=dict(lw=1, capsize=2, ecolor="#333333"))
        ax.set_yticks(list(y), labels=[f"{l}  " for l in labels], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIG_DIR, "unified_h10_metrics.png")
    fig.savefig(out, dpi=200)
    print(f"wrote {out} ({len(reps)} representative rows)")


def fig_budget(rows: dict[str, dict]) -> None:
    hist = sorted(
        (int(r["step"]), r) for run, r in rows.items()
        if run.startswith("act_umi_identity_rot6d_1459_")
    )
    r50 = sorted(
        (int(r["step"]), r) for run, r in rows.items()
        if run.startswith("act_r50_v1_vae_seed1000_")
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Unified horizon-10 budget curves — accuracy@0.1 and XYZ endpoint vs training steps",
        fontsize=12,
    )
    specs = [
        (axes[0], "action_acc_at_0p1", "accuracy@0.1 (action, normalized)", 1, (0.86, 0.95)),
        (axes[1], "xyz_end_m", "XYZ endpoint error (mm)", 1000, (8, 17)),
    ]
    for ax, met, ylab, scale, ylim in specs:
        for series, color, label, marker in (
            (hist, COLORS["hist"], "ACT R18-VAE (historical, 30 ckpts)", "o"),
            (r50, COLORS["r50v1"], "ACT R50-V1 (fresh 1M run)", "s"),
        ):
            if not series:
                continue
            xs = [s for s, _ in series]
            ys = [float(r[met]) * scale for _, r in series]
            ax.plot(xs, ys, marker=marker, ms=4, lw=1.5, color=color, label=label)
            lo = series[0][1].get(f"{met}__ci_low")
            if lo:  # bootstrap band (deterministic across renders)
                ax.fill_between(
                    xs,
                    [float(r[f"{met}__ci_low"]) * scale for _, r in series],
                    [float(r[f"{met}__ci_high"]) * scale for _, r in series],
                    color=color, alpha=0.12, lw=0,
                )
        # single-budget references; per-panel label offsets (points) — the
        # acc@0.1 panel clusters several references at x=20k, y≈0.92 and needs
        # a wider vertical stack than the mm panel.
        dy0, dy_step = (14, 9) if met == "action_acc_at_0p1" else (8, 6)
        for i, (run, label, fam) in enumerate((
            ("pi05_lora_sroi_rot6d_seed1000_0020000steps", "openpi rot6d", "openpi"),
            ("pi05_lora_sroi_rotvec_seed1000_0020000steps", "openpi rotvec", "openpi"),
            ("pi05_lora_sroi_rot6d_h30_seed1000_0020000steps", "openpi h30", "openpi"),
            ("pi05_port_openpi_recipe_seed1000_020000steps", "π0.5 port o-recipe", "port"),
            ("act_r18_flow_u_lr1e5_seed2000_100000steps", "ACT-flow 50k", "flow"),
            ("act_r18_l1_seed2000_100000steps", "ACT-L1 100k", "actl1"),
            ("act_r50_vae_seed2000_100000steps", "R50-VAE 80k", "r50vae"),
            ("pi05_port_seed1000_1000000steps", "π0.5 port 1M", "port"),
            ("smolvla_rot6d_seed1000_100000steps", "SmolVLA rot6d", "smol"),
        )):
            if run not in rows:
                continue
            r = rows[run]
            x, y = int(r["step"]), float(r[met]) * scale
            ax.scatter([x], [y], color=COLORS[fam], marker="*", s=110, zorder=5)
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(4, dy0 + i * dy_step), fontsize=7, color=COLORS[fam])
        ax.set_xscale("log")
        ax.set_xlabel("training steps")
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(FIG_DIR, "unified_h10_budget.png")
    fig.savefig(out, dpi=200)
    print(f"wrote {out} ({len(hist)} hist + {len(r50)} R50-V1 points)")


def main() -> int:
    rows = load_rows()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_metrics(rows)
    fig_budget(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

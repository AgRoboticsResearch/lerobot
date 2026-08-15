#!/usr/bin/env python3
"""Figures for the rotation-notation ablation (§9.2.3) and the official-openpi
replication (§9.2.4).

Figure A (notation_cross_stack): rot6d vs axis-angle/rotvec across BOTH stacks
(SmolVLA PyTorch, openpi π0.5 JAX) -- shows the endpoint-accuracy tie and the
stack-specific sign flip of the small jitter effects.

Figure B (openpi_budget_context): endpoint XYZ error vs training steps (log x)
for the main trained families -- the official-openpi 20k-step points versus the
ACT matrix, the SmolVLA notation runs, and the lerobot-port π0.5 reference --
the "official recipe >> port at 1/35th budget" headline.

Reads JSONs produced by eval_open_loop_dataset.py / eval_openpi_open_loop.py
plus the stage1 CSV for ACT R50-V1. Outputs into act_flow_ablation/figures/.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/data1/projects/lerobot-arch-exp")
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "svg.hashsalt": "umi-notation-openpi",
})

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def load_eb(path: Path) -> tuple[dict, dict]:
    d = json.load(open(path))
    s = d["summary"]
    return s["episode_balanced"], s.get("episode_balanced_95ci", {})


def mci(eb: dict, ci: dict, key: str, sf: float) -> tuple[float, float, float]:
    return eb[key] * sf, ci[key]["low"] * sf, ci[key]["high"] * sf


# ---------------------------------------------------------------- data ---- #
SMOL = ROOT / "outputs/research_report/smolvla_notation_eval_20260814"
OPENPI = ROOT / "outputs/research_report/openpi_sroi_eval"
PI05 = ROOT / "eval_common_h32"

smol_r6 = load_eb(SMOL / "smolvla_rot6d_seed1000_100000steps_100000_open_loop_metrics.json")
smol_aa = load_eb(SMOL / "smolvla_axis_angle_seed1000_100000steps_100000_open_loop_metrics.json")
op_rv = load_eb(OPENPI / "pi05_lora_sroi_rotvec_final_open_loop_metrics.json")
op_r6 = load_eb(OPENPI / "pi05_lora_sroi_rot6d_final_open_loop_metrics.json")
pi05_700 = load_eb(PI05 / "pi05_openpi_split_lora_1459_700k/seed1000/"
                   "pi05_openpi_split_lora_masked_1459_bs4_1m_0700000_open_loop_metrics.json")

# ACT R50-V1 100k from the stage-1 summary CSV
act = None
with open(ROOT / "lerobot-arch-exp/results/stage1_variant_summary.csv") as f:
    for row in csv.DictReader(f):
        if row["variant"] == "act_r50_v1_vae" and row["steps"] == "100000":
            act = row
            break
assert act is not None, "ACT R50-V1 100k row not found"

# ------------------------------------------------- figure A: notation ---- #
C_R6 = "#1F5A94"     # blue
C_AA = "#E17C05"     # orange
panels = [
    ("XYZ endpoint error (mm)", "xyz_end_m", 1000.0),
    ("Rotation endpoint error (deg)", "rotation_end_deg", 1.0),
    ("Rotation jitter (deg)", "rot_jerk_deg", 1.0),
]
stacks = [
    ("SmolVLA (PyTorch)", [("rot6d", smol_r6), ("axis-angle", smol_aa)]),
    ("openpi pi0.5 LoRA (JAX)", [("rot6d", op_r6), ("rotvec", op_rv)]),
]

fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
for ax, (title, key, sf) in zip(axes, panels):
    xpos, labels, colors = [], [], []
    x = 0.0
    for si, (stack, arms) in enumerate(stacks):
        for ai, (name, (eb, ci)) in enumerate(arms):
            m, lo, hi = mci(eb, ci, key, sf)
            col = C_R6 if name == "rot6d" else C_AA
            ax.bar(x, m, width=0.72, color=col, zorder=3)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none",
                        ecolor=INK, elinewidth=1.2, capsize=3, zorder=4)
            labels.append(name)
            colors.append(col)
            xpos.append(x)
            x += 1.0
        x += 0.7  # gap between stacks
    ymax = max(
        mci(eb, ci, key, sf)[2]
        for _, arms in stacks
        for _, (eb, ci) in arms
    )
    ax.set_ylim(0, ymax * 1.22)
    for si, (stack, _) in enumerate(stacks):
        x0 = si * 2.7 - 0.55
        ax.axvline(x0 + 2.6, color=GRID, lw=0.8, zorder=1)
        ax.text(x0 + 1.0, ymax * 1.13, stack, ha="center", va="top",
                fontsize=9, color=MUTED)
    if key == "rot_jerk_deg":
        gt = smol_r6[0]["gt_rot_jerk_deg"]
        ax.axhline(gt, color=MUTED, lw=1.0, ls="--", zorder=2)
        ax.text(3.9, gt + ymax * 0.015, "GT", fontsize=8, color=MUTED,
                va="bottom", ha="right")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title(title, fontsize=10.5)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.tick_params(labelsize=8.5)
fig.suptitle("Rotation notation: endpoint accuracy ties on both stacks; "
             "small jitter effects flip sign with the stack", fontsize=11, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "notation_cross_stack.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "notation_cross_stack.svg", bbox_inches="tight")
plt.close(fig)

# ------------------------------------- figure B: openpi budget context ---- #
series = [
    # label, steps, (eb, ci), color, marker
    ("ACT R50-V1 (100k, L1)", 100_000,
     ({"xyz_end_m": float(act["xyz_end_m"])},
      {"xyz_end_m": {"low": float(act["xyz_end_m_ci_low"]), "high": float(act["xyz_end_m_ci_high"])}}),
     "#59A14F", "s"),
    ("SmolVLA rot6d (100k)", 100_000, smol_r6, "#54A24B", "^"),
    ("pi0.5 LoRA port (700K)", 700_000, pi05_700, "#7A5195", "D"),
    ("openpi pi0.5 rotvec (20k)", 20_000, op_rv, "#E45756", "o"),
    ("openpi pi0.5 rot6d (20k)", 20_000, op_r6, "#B33B3A", "o"),
]
fig, ax = plt.subplots(figsize=(7.6, 4.6))
for label, steps, (eb, ci), color, marker in series:
    m, lo, hi = mci(eb, ci, "xyz_end_m", 1000.0)
    ax.errorbar(steps, m, yerr=[[m - lo], [hi - m]], fmt=marker, ms=8, mfc=color,
                mec="white", mew=1.2, ecolor=color, elinewidth=1.4, capsize=3,
                label=label, zorder=3)
    ax.annotate(label, (steps, m), textcoords="offset points", xytext=(8, 6),
                fontsize=8.5, color=INK)
ax.set_xscale("log")
ax.set_xlabel("Training steps (log scale)")
ax.set_ylabel("Endpoint XYZ error (mm), 95% CI")
ax.grid(color=GRID, lw=0.7, zorder=0)
ax.set_xlim(1.2e4, 2.2e6)
ax.set_ylim(bottom=0)
ax.set_title("Official openpi at 20k steps beats every longer-budget run\n"
             "(same fixed 100-episode / 500-query validation protocol)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "openpi_budget_context.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "openpi_budget_context.svg", bbox_inches="tight")
plt.close(fig)

print(f"wrote figures to {OUT}")
for f in sorted(OUT.glob("notation_*")) + sorted(OUT.glob("openpi_budget*")):
    print(" ", f.name)

#!/usr/bin/env python3
"""Figures for the rotation-notation ablation (§9.2.3) and the official-openpi
replication (§9.2.4).

Figure A (notation_cross_stack): rot6d vs axis-angle/rotvec across BOTH stacks
(SmolVLA PyTorch, openpi π0.5 JAX) -- shows the endpoint-accuracy tie and the
stack-specific sign flip of the small jitter effects.

Figure B (openpi_budget_context): HORIZON-MATCHED (10-step) endpoint XYZ error
vs training samples seen for SmolVLA / the lerobot-port π0.5 (both re-scored
with --eval_horizon 10) and the official-openpi arms. Corrected 2026-08-16:
the original cross-horizon version of this figure (endpoint@t+10 vs endpoint@t+30)
showed a spurious "2.3x openpi lead"; at matched horizon all stacks tie and the
real openpi advantage is sample efficiency (~9x fewer samples).

Reads JSONs produced by eval_open_loop_dataset.py / eval_openpi_open_loop.py;
the h10 re-scores live under outputs/research_report/h10_matched_eval/.
Outputs into act_flow_ablation/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

smol_r6 = load_eb(SMOL / "smolvla_rot6d_seed1000_100000steps_100000_open_loop_metrics.json")
smol_aa = load_eb(SMOL / "smolvla_axis_angle_seed1000_100000steps_100000_open_loop_metrics.json")
op_rv = load_eb(OPENPI / "pi05_lora_sroi_rotvec_final_open_loop_metrics.json")
op_r6 = load_eb(OPENPI / "pi05_lora_sroi_rot6d_final_open_loop_metrics.json")

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

# ------------------------- figure B: horizon-matched cross-stack view ---- #
# Corrected (2026-08-16): the original budget-context figure compared endpoint
# errors across DIFFERENT chunk horizons (10 vs 30 steps) -- a confound that
# produced a spurious "2.3x openpi lead". This version scores everything at the
# same 10-step horizon (port/SmolVLA re-scored via --eval_horizon 10) and plots
# endpoint vs samples seen: the corrected story is parity of accuracy + the
# openpi recipe's sample efficiency.
PORT_H10 = ROOT / "outputs/research_report/h10_matched_eval"
PORT_SEEDS = PORT_H10 / "port_seeds"  # seed{1000,2000,3000}.json


def load_seed_mean(pattern: str) -> tuple[dict, dict]:
    files = sorted((PORT_SEEDS if "seed" in pattern else PORT_H10).glob(pattern))
    ebs = [json.load(open(f))["summary"]["episode_balanced"] for f in files]
    keys = ebs[0].keys()
    mean = {k: float(np.mean([e[k] for e in ebs])) for k in keys}
    ci = {}
    for k in keys:
        los = [json.load(open(f))["summary"]["episode_balanced_95ci"][k]["low"] for f in files]
        his = [json.load(open(f))["summary"]["episode_balanced_95ci"][k]["high"] for f in files]
        ci[k] = {"low": float(np.mean(los)), "high": float(np.mean(his))}
    return mean, ci


port_h10 = load_seed_mean("seed*.json")
smol_h10 = load_seed_mean("smolvla_h10.json")

series = [
    # label, samples seen, (eb, ci), color, marker
    ("SmolVLA rot6d 100k @800k samples", 800_000, smol_h10, "#54A24B", "^"),
    ("pi0.5 port 700K @2.8M samples", 2_800_000, port_h10, "#7A5195", "D"),
    ("openpi rot6d 20k @320k samples", 320_000, op_r6, "#B33B3A", "o"),
    ("openpi rotvec 20k @320k samples", 320_000, op_rv, "#E45756", "o"),
]
fig, ax = plt.subplots(figsize=(7.8, 4.8))
for label, samples, (eb, ci), color, marker in series:
    m, lo, hi = mci(eb, ci, "xyz_end_m", 1000.0)
    ax.errorbar(samples, m, yerr=[[m - lo], [hi - m]], fmt=marker, ms=9, mfc=color,
                mec="white", mew=1.2, ecolor=color, elinewidth=1.4, capsize=3,
                label=label, zorder=3)
    ax.annotate(label, (samples, m), textcoords="offset points", xytext=(10, 7),
                fontsize=8.5, color=INK)
ax.set_xscale("log")
ax.set_xlabel("Training samples seen (log scale)")
ax.set_ylabel("Endpoint XYZ error at t+10 (mm), 95% CI")
ax.grid(color=GRID, lw=0.7, zorder=0)
ax.set_xlim(1.8e5, 5e6)
ax.set_ylim(0, 13)
ax.set_title("Horizon-matched (10-step) endpoint: all stacks statistically tied;\n"
             "official openpi reaches it with ~9x fewer samples", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "openpi_budget_context.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "openpi_budget_context.svg", bbox_inches="tight")
plt.close(fig)

print(f"wrote figures to {OUT}")
for f in sorted(OUT.glob("notation_*")) + sorted(OUT.glob("openpi_budget*")):
    print(" ", f.name)

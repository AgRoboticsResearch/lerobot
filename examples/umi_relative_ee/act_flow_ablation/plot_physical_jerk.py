#!/usr/bin/env python3
"""Figures for the §9.2.13 physical-jerk re-evaluation.

Reads the compiler's physical_jerk_h10.csv (produced by
compile_physical_jerk.py from the kiwi re-eval tree; legacy metrics
cross-validated there against the archived §9.2.9 numbers) and renders:

  figures/physical_jerk_h10.png       — representative bars of true rot
    jerk (deg/s³) and XYZ jerk (mm/s³) at dt = 1/30 s, 95% episode
    bootstrap CIs, GT reference lines.
  figures/physical_jerk_ratio.png      — pred/GT ratio ladder across
    velocity → acceleration → jerk (rot and XYZ): where in the derivative
    stack each family's smoothing/jitter signature appears.
  figures/physical_jerk_budget.png     — physical jerk vs training steps
    for the two budget curves plus single-budget stars, GT reference.

The four JAX openpi rows are physical-pending (host re-eval after the h30
training frees the GPU) and are skipped here. Run via:
  /home/zfei/anaconda3/envs/py312/bin/python \
    examples/umi_relative_ee/act_flow_ablation/plot_physical_jerk.py
"""
from __future__ import annotations

import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = "/mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/results_physical_jerk/physical_jerk_h10.csv"
FIG_DIR = os.path.join(HERE, "figures")

COLORS = {
    "hist": "#444444",
    "r50v1": "#1f77b4",
    "actl1": "#2ca02c",
    "r50vae": "#9467bd",
    "flow": "#d62728",
    "port": "#ff7f0e",
    "openpi": "#17becf",
    "smol": "#8c564b",
    "smol1m": "#e377c2",
    "smolmask": "#7f7f7f",
    "openpi1m": "#bcbd22",
}

# Same representatives as the §9.2.9 figures (torch families only here).
REPRESENTATIVES = [
    ("act_umi_identity_rot6d_1459_3000000steps", "ACT R18-VAE 3M (hist)", "hist"),
    ("act_r50_v1_vae_seed1000_0800000steps", "ACT R50-V1 800k", "r50v1"),
    ("act_r18_l1_seed2000_100000steps", "ACT-L1 100k s2000", "actl1"),
    ("act_r18_l1_seed3000_100000steps", "ACT-L1 100k s3000", "actl1"),
    ("act_r50_vae_seed2000_100000steps", "ACT R50-VAE 80k s2000", "r50vae"),
    ("act_r50_vae_seed3000_100000steps", "ACT R50-VAE 80k s3000", "r50vae"),
    ("act_r18_flow_u_lr1e5_seed2000_100000steps", "ACT-flow 50k s2000", "flow"),
    ("act_r18_flow_u_lr1e5_seed3000_100000steps", "ACT-flow 50k s3000", "flow"),
    ("pi05_port_openpi_recipe_seed1000_020000steps", "π0.5 port o-recipe 20k", "port"),
    ("pi05_port_seed1000_1000000steps", "π0.5 port 1M (kiwi)", "port"),
    ("pi05_port_seed1000_0700000steps", "π0.5 port 700K (kiwi)", "port"),
    ("smolvla_rot6d_seed1000_100000steps", "SmolVLA rot6d 100k", "smol"),
    ("smolvla_rot6d_seed1000_1000000steps", "SmolVLA rot6d 1M", "smol1m"),
    ("smolvla_axis_angle_seed1000_100000steps", "SmolVLA axis-ang 100k", "smol"),
    ("smolvla_masked_seed1000_1000000steps", "SmolVLA masked 1M", "smolmask"),
]

FAMILY_PREFIXES = [
    ("act_umi_identity_rot6d_1459_", "hist", "R18-VAE hist"),
    ("act_r50_v1_vae_", "r50v1", "R50-V1"),
    ("act_r18_l1_", "actl1", "ACT-L1"),
    ("act_r50_vae_", "r50vae", "R50-VAE"),
    ("act_r18_flow_u_lr1e5_", "flow", "ACT-flow"),
    ("pi05_port_openpi_recipe_", "port", "π0.5 port o-recipe"),
    ("pi05_lora_sroi_", "openpi", "openpi (pending)"),
    ("pi05_openpi1m_", "openpi1m", "openpi (pending)"),
    ("pi05_port_", "port", "π0.5 port"),
    ("smolvla_rot6d_", "smol", "SmolVLA rot6d"),
    ("smolvla_axis_angle_", "smol", "SmolVLA axis-angle"),
    ("smolvla_masked_", "smolmask", "SmolVLA masked"),
]


def family_of(run: str) -> tuple[str, str]:
    for prefix, fam, label in FAMILY_PREFIXES:
        if run.startswith(prefix):
            return fam, label
    return "hist", run


def fmt_step(step: int) -> str:
    if step % 1_000_000 == 0:
        return f"{step // 1_000_000}M"
    return f"{step // 1000}k"


def load_rows() -> dict[str, dict]:
    with open(CSV_PATH) as f:
        rows = {r["run"]: r for r in csv.DictReader(f)}
    return {k: v for k, v in rows.items() if v.get("rot_jerk_deg_s3") not in ("nan", "", None)}


def have(rows: dict[str, dict], run: str) -> bool:
    return run in rows and not math.isnan(float(rows[run]["rot_jerk_deg_s3"]))


def fig_jerk_bars(rows: dict[str, dict]) -> None:
    reps = [(r, lab, fam) for r, lab, fam in REPRESENTATIVES if have(rows, r)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    panels = [
        ("rot", "rot_jerk_deg_s3", "gt_rot_jerk_deg_s3", "Rotational jerk (deg/s³)"),
        ("xyz", "xyz_jerk_mm_s3", "gt_xyz_jerk_mm_s3", "XYZ jerk (mm/s³)"),
    ]
    for ax, (_, m, gm, title) in zip(axes, panels):
        xs = range(len(reps))
        vals = [float(rows[r][m]) for r, _, _ in reps]
        los = [float(rows[r][m]) - float(rows[r][f"{m}_lo"]) for r, _, _ in reps]
        his = [float(rows[r][f"{m}_hi"]) - float(rows[r][m]) for r, _, _ in reps]
        colors = [COLORS[fam] for _, _, fam in reps]
        ax.bar(xs, vals, yerr=[los, his], color=colors, capsize=3, error_kw={"lw": 1})
        gt = float(rows[reps[0][0]][gm])
        ax.axhline(gt, color="k", ls="--", lw=1.2, label=f"GT ({gt:,.0f})" if m.startswith("rot") else f"GT ({gt:,.0f})")
        ax.set_xticks(list(xs), [lab for _, lab, _ in reps], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{title}  —  h10 chunk, dt = 1/30 s")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "physical_jerk_h10.png")
    fig.savefig(out, dpi=200)
    print(out)


def fig_ratio_ladder(rows: dict[str, dict]) -> None:
    reps = [(r, lab, fam) for r, lab, fam in REPRESENTATIVES if have(rows, r)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ladders = [
        (["rot_vel_deg_s", "rot_accel_deg_s2", "rot_jerk_deg_s3"],
         ["gt_rot_vel_deg_s", "gt_rot_accel_deg_s2", "gt_rot_jerk_deg_s3"],
         ["vel", "accel", "jerk"], "Rotation: pred/GT ratio"),
        (["xyz_vel_mm_s", "xyz_accel_mm_s2", "xyz_jerk_mm_s3"],
         ["gt_xyz_vel_mm_s", "gt_xyz_accel_mm_s2", "gt_xyz_jerk_mm_s3"],
         ["vel", "accel", "jerk"], "XYZ: pred/GT ratio"),
    ]
    mm = {}  # CSV stores gt xyz columns already in mm
    for ax, (ms, gms, ticks, title) in zip(axes, ladders):
        w = 0.26
        for i, (run, _, fam) in enumerate(reps):
            for j, (m, gm) in enumerate(zip(ms, gms)):
                ratio = float(rows[run][m]) / (float(rows[run][gm]) * mm.get(gm, 1.0))
                ax.bar(i + (j - 1) * w, ratio, width=w, color=COLORS[fam], alpha=1.0 - 0.25 * j)
        ax.axhline(1.0, color="k", ls="--", lw=1.2)
        ax.set_xticks(range(len(reps)), [lab for _, lab, _ in reps], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{title}  (<1 over-smooths, >1 jitters)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(handles=[mpatches.Rectangle((0, 0), 1, 1, alpha=1.0 - 0.25 * j) for j in range(3)],
                  labels=ticks, fontsize=9, title="derivative")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "physical_jerk_ratio.png")
    fig.savefig(out, dpi=200)
    print(out)


def fig_budget(rows: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    panels = [
        ("rot_jerk_deg_s3", "Rotational jerk (deg/s³)"),
        ("xyz_jerk_mm_s3", "XYZ jerk (mm/s³)"),
    ]
    curves = [
        ("act_umi_identity_rot6d_1459_", "hist", "R18-VAE hist"),
        ("act_r50_v1_vae_", "r50v1", "R50-V1"),
    ]
    singles = [
        ("act_r18_l1_", "actl1", "ACT-L1"),
        ("act_r50_vae_", "r50vae", "R50-VAE"),
        ("act_r18_flow_u_lr1e5_", "flow", "ACT-flow"),
        ("pi05_port_", "port", "π0.5 port"),
        ("smolvla_", "smol", "SmolVLA"),
    ]
    for ax, (m, title) in zip(axes, panels):
        gt = float(next(iter(rows.values()))[f"gt_{m}"])
        for prefix, fam, label in curves:
            pts = sorted((int(r["steps"]), float(r[m])) for run, r in rows.items()
                         if run.startswith(prefix) and "seed" not in run)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=3, color=COLORS[fam], label=label)
        for prefix, fam, label in singles:
            pts = [(int(r["steps"]), float(r[m])) for run, r in rows.items() if run.startswith(prefix)]
            if pts:
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker="*", s=60,
                           color=COLORS[fam], label=label, zorder=5)
        ax.axhline(gt, color="k", ls="--", lw=1.2, label=f"GT ({gt:,.0f})")
        ax.set_xlabel("training steps")
        ax.set_title(f"{title} vs training budget")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "physical_jerk_budget.png")
    fig.savefig(out, dpi=200)
    print(out)


def main() -> int:
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = load_rows()
    print(f"{len(rows)} runs with physical metrics")
    fig_jerk_bars(rows)
    fig_ratio_ladder(rows)
    fig_budget(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

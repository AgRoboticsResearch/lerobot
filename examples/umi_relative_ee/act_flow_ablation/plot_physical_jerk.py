#!/usr/bin/env python3
"""Figures for the §9.2.13 physical-dynamics re-evaluation.

Reads the compiler's physical_jerk_h10.csv (produced by
compile_physical_jerk.py from the kiwi re-eval tree; legacy metrics
cross-validated there against the archived §9.2.9 numbers) and renders:

  figures/physical_{velocity,acceleration,jerk}_h10.png — representative
    rotational and XYZ bars at dt = 1/30 s, 95% episode bootstrap CIs,
    GT reference lines.
  figures/physical_{velocity,acceleration,jerk}_all.png — each derivative
    for EVERY run of the sweep (log-scale horizontal bars, family-grouped,
    bootstrap CIs, dashed GT).
  figures/physical_jerk_ratio.png      — pred/GT ratio ladder across
    velocity → acceleration → jerk (rot and XYZ): where in the derivative
    stack each family's smoothing/jitter signature appears.
  figures/physical_{velocity,acceleration,jerk}_budget.png — each physical
    derivative vs training steps for budget curves plus single-budget stars.
  figures/physical_dynamics_budget.png — the same six budget panels together:
    rotational velocity/acceleration/jerk above XYZ velocity/acceleration/jerk.

The four JAX openpi rows are physical-pending (host re-eval after the h30
training frees the GPU) and are skipped here. Run via:
  /home/zfei/anaconda3/envs/py312/bin/python \
    examples/umi_relative_ee/act_flow_ablation/plot_physical_jerk.py
"""
from __future__ import annotations

import csv
import math
import os
import re

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
    "2frame": "#1f9e89",
    "noproprio": "#8c564b",
    "state3": "#bcbd22",
    "state5": "#e6550d",
    "state10": "#756bb1",
    "actl1": "#2ca02c",
    "r50vae": "#9467bd",
    "flow": "#d62728",
    "port": "#ff7f0e",
    "openpi": "#17becf",
    "smol": "#8c564b",
    "smol1m": "#e377c2",
    "smolmask": "#7f7f7f",
    "openpi1m": "#bcbd22",
    "lingbot": "#ff9896",
}

# Same representatives as the §9.2.9 figures (torch families only here).
REPRESENTATIVES = [
    ("act_umi_identity_rot6d_1459_3000000steps", "ACT R18-VAE 3M (hist)", "hist"),
    ("act_r50_v1_vae_seed1000_0800000steps", "ACT R50-VAE (ImageNet-V1) 800k", "r50v1"),
    ("act_r50_v1_vae_2frame_seed1000_0500000steps", "ACT R50-VAE 2-frame 500k (Q3)", "2frame"),
    ("act_r50_v1_vae_noproprio_seed1000_0500000steps", "ACT R50-VAE no-proprio 500k (Q4)", "noproprio"),
    ("act_r50_v1_vae_state3_seed1000_0500000steps", "ACT R50-VAE state-W3 500k (Q4+)", "state3"),
    ("act_r50_v1_vae_state5_seed1000_0500000steps", "ACT R50-VAE state-W5 500k (Q4+)", "state5"),
    ("act_r50_v1_vae_state10_seed1000_0500000steps", "ACT R50-VAE state-W10 500k (Q4+)", "state10"),
    ("act_r18_l1_seed2000_100000steps", "ACT-L1 100k s2000", "actl1"),
    ("act_r18_l1_seed3000_100000steps", "ACT-L1 100k s3000", "actl1"),
    ("act_r50_vae_seed2000_100000steps", "ACT R50-VAE (ImageNet-V2) 80k s2000", "r50vae"),
    ("act_r50_vae_seed3000_100000steps", "ACT R50-VAE (ImageNet-V2) 80k s3000", "r50vae"),
    ("act_r18_flow_u_lr1e5_seed2000_100000steps", "ACT-flow 50k s2000", "flow"),
    ("act_r18_flow_u_lr1e5_seed3000_100000steps", "ACT-flow 50k s3000", "flow"),
    ("pi05_port_openpi_recipe_seed1000_020000steps", "π0.5 port o-recipe 20k", "port"),
    ("pi05_port_seed1000_1000000steps", "π0.5 port 1M (kiwi)", "port"),
    ("pi05_port_seed1000_0700000steps", "π0.5 port 700K (kiwi)", "port"),
    ("smolvla_rot6d_seed1000_100000steps", "SmolVLA rot6d 100k", "smol"),
    ("smolvla_rot6d_seed1000_1000000steps", "SmolVLA rot6d 1M", "smol1m"),
    ("smolvla_axis_angle_seed1000_100000steps", "SmolVLA axis-ang 100k", "smol"),
    ("smolvla_masked_seed1000_1000000steps", "SmolVLA masked 1M", "smolmask"),
    ("lingbot_va_axis_angle_seed1000_200000steps", "LingBot-VA 200k", "lingbot"),
]

FAMILY_PREFIXES = [
    ("act_umi_identity_rot6d_1459_", "hist", "R18-VAE hist"),
    # 2frame/noproprio/state3/state5/state10 prefixes MUST precede act_r50_v1_vae_ (strict extensions).
    ("act_r50_v1_vae_noproprio_", "noproprio", "R50-VAE no-proprio (Q4)"),
    ("act_r50_v1_vae_2frame_", "2frame", "R50-VAE 2-frame (Q3)"),
    ("act_r50_v1_vae_state10_", "state10", "R50-VAE state-W10 (Q4+)"),
    ("act_r50_v1_vae_state5_", "state5", "R50-VAE state-W5 (Q4+)"),
    ("act_r50_v1_vae_state3_", "state3", "R50-VAE state-W3 (Q4+)"),
    ("act_r50_v1_vae_", "r50v1", "R50-VAE (ImageNet-V1)"),
    ("act_r18_l1_", "actl1", "ACT-L1"),
    ("act_r50_vae_", "r50vae", "R50-VAE (ImageNet-V2)"),
    ("act_r18_flow_u_lr1e5_", "flow", "ACT-flow"),
    ("pi05_port_openpi_recipe_", "port", "π0.5 port o-recipe"),
    ("pi05_lora_sroi_", "openpi", "openpi (pending)"),
    ("pi05_openpi1m_", "openpi1m", "openpi (pending)"),
    ("pi05_port_", "port", "π0.5 port"),
    ("smolvla_rot6d_", "smol", "SmolVLA rot6d"),
    ("smolvla_axis_angle_", "smol", "SmolVLA axis-angle"),
    ("smolvla_masked_", "smolmask", "SmolVLA masked"),
    ("lingbot_va_axis_angle_", "lingbot", "LingBot-VA"),
]


def family_of(run: str) -> tuple[str, str] | None:
    for prefix, fam, label in FAMILY_PREFIXES:
        if run.startswith(prefix):
            return fam, label
    return None


FAMILY_ORDER = [fam for _, fam, _ in FAMILY_PREFIXES]

DYNAMICS = {
    "velocity": (
        ("rot_vel_deg_s", "gt_rot_vel_deg_s", "Rotational velocity (deg/s)"),
        ("xyz_vel_mm_s", "gt_xyz_vel_mm_s", "XYZ velocity (mm/s)"),
    ),
    "acceleration": (
        ("rot_accel_deg_s2", "gt_rot_accel_deg_s2", "Rotational acceleration (deg/s²)"),
        ("xyz_accel_mm_s2", "gt_xyz_accel_mm_s2", "XYZ acceleration (mm/s²)"),
    ),
    "jerk": (
        ("rot_jerk_deg_s3", "gt_rot_jerk_deg_s3", "Rotational jerk (deg/s³)"),
        ("xyz_jerk_mm_s3", "gt_xyz_jerk_mm_s3", "XYZ jerk (mm/s³)"),
    ),
}


def _seed_of(run: str) -> str:
    m = re.search(r"_seed(\d+)_", run)
    return m.group(1) if m else "1000"


def all_runs_label(run: str, row: dict) -> str:
    _, base = family_of(run) or ("hist", run)
    seed = _seed_of(run)
    s = f" s{seed}" if seed != "1000" else ""
    return f"{base}{s} {fmt_step(int(row['steps']))}"


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


def fig_derivative_bars(rows: dict[str, dict], derivative: str) -> None:
    reps = [(r, lab, fam) for r, lab, fam in REPRESENTATIVES if have(rows, r)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, (m, gm, title) in zip(axes, DYNAMICS[derivative], strict=True):
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
    out = os.path.join(FIG_DIR, f"physical_{derivative}_h10.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
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
    for ax, (ms, gms, ticks, title) in zip(axes, ladders, strict=True):
        w = 0.26
        for i, (run, _, fam) in enumerate(reps):
            for j, (m, gm) in enumerate(zip(ms, gms, strict=True)):
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
    plt.close(fig)
    print(out)


def _budget_series():
    curves = [
        (lambda run: run.startswith("act_umi_identity_rot6d_1459_"), "hist", "R18-VAE hist", "o"),
        (lambda run: run.startswith("act_r50_v1_vae_seed1000_"), "r50v1", "R50-VAE (ImageNet-V1)", "s"),
        (lambda run: run.startswith("act_r50_v1_vae_2frame_seed1000_"), "2frame", "R50-VAE 2-frame (Q3)", "x"),
        (lambda run: run.startswith("act_r50_v1_vae_noproprio_seed1000_"), "noproprio", "R50-VAE no-proprio (Q4)", "P"),
        (lambda run: run.startswith("act_r50_v1_vae_state3_seed1000_"), "state3", "R50-VAE state-W3 (Q4+)", "+"),
        (lambda run: run.startswith("act_r50_v1_vae_state5_seed1000_"), "state5", "R50-VAE state-W5 (Q4+)", "<"),
        (lambda run: run.startswith("act_r50_v1_vae_state10_seed1000_"), "state10", "R50-VAE state-W10 (Q4+)", ">"),
        (lambda run: run.startswith("pi05_port_seed1000_"), "port", "π0.5 port", "^"),
        (lambda run: re.fullmatch(r"smolvla_rot6d_seed1000_\d{7}steps", run) is not None,
         "smol1m", "SmolVLA full-width", "D"),
        (lambda run: re.fullmatch(r"smolvla_masked_seed1000_\d{7}steps", run) is not None,
         "smolmask", "SmolVLA masked", "v"),
        (lambda run: run.startswith("lingbot_va_axis_angle_seed1000_"),
         "lingbot", "LingBot-VA", "*"),
    ]
    singles = [
        ("act_r18_l1_", "actl1", "ACT-L1"),
        ("act_r50_vae_", "r50vae", "R50-VAE (ImageNet-V2)"),
        ("act_r18_flow_u_lr1e5_", "flow", "ACT-flow"),
        ("pi05_port_openpi_recipe_", "port", "π0.5 port o-recipe"),
        ("smolvla_rot6d_seed1000_100000steps", "smol", "SmolVLA rot6d short"),
        ("smolvla_axis_angle_", "smol", "SmolVLA axis-angle short"),
    ]
    return curves, singles


def _plot_budget_axis(ax, rows: dict[str, dict], metric: str, title: str) -> None:
    curves, singles = _budget_series()
    gt = float(next(iter(rows.values()))[f"gt_{metric}"])
    for select, fam, label, marker in curves:
        selected = sorted((int(r["steps"]), r) for run, r in rows.items() if select(run))
        if selected:
            xs = [step for step, _ in selected]
            ax.plot(
                xs,
                [float(r[metric]) for _, r in selected],
                marker=marker,
                ms=3,
                lw=1.5,
                color=COLORS[fam],
                label=label,
            )
            ax.fill_between(
                xs,
                [float(r[f"{metric}_lo"]) for _, r in selected],
                [float(r[f"{metric}_hi"]) for _, r in selected],
                color=COLORS[fam],
                alpha=0.10,
                lw=0,
            )
    for prefix, fam, label in singles:
        pts = [
            (int(r["steps"]), float(r[metric]), float(r[f"{metric}_lo"]), float(r[f"{metric}_hi"]))
            for run, r in rows.items()
            if run.startswith(prefix)
        ]
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.errorbar(
                xs,
                ys,
                yerr=[[y - lo for _, y, lo, _ in pts], [hi - y for _, y, _, hi in pts]],
                fmt="*",
                ms=8,
                capsize=2,
                lw=0.8,
                color=COLORS[fam],
                label=label,
                zorder=5,
            )
    ax.axhline(gt, color="k", ls="--", lw=1.2, label=f"GT ({gt:,.0f})")
    ax.set_xscale("log")
    ax.set_xlabel("training steps")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def fig_derivative_budget(rows: dict[str, dict], derivative: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, (metric, _, title) in zip(axes, DYNAMICS[derivative], strict=True):
        _plot_budget_axis(ax, rows, metric, f"{title} vs training budget")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    out = os.path.join(FIG_DIR, f"physical_{derivative}_budget.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(out)


def fig_dynamics_budget(rows: dict[str, dict]) -> None:
    """Put all six physical budget panels in the same layout as concise Fig. 15."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for col, derivative in enumerate(DYNAMICS):
        for row, (metric, _, title) in enumerate(DYNAMICS[derivative]):
            _plot_budget_axis(axes[row, col], rows, metric, title)
    fig.suptitle(
        "Canonical-h10 physical motion dynamics vs training budget (dt = 1/30 s)",
        fontsize=14,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))
    out = os.path.join(FIG_DIR, "physical_dynamics_budget.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(out)


def fig_derivative_all(rows: dict[str, dict], derivative: str) -> None:
    """One physical derivative for EVERY run of the sweep, in a shared format.

    Mirrors §9.2.9's within-chunk second-difference figure
    (unified_h10_jitter.png): one row per run, grouped by family (budget
    curves first, single-budget references after), 95% episode bootstrap
    CIs, dashed GT line per panel — but in physical units at dt = 1/30 s.
    """
    entries = []
    for run, r in rows.items():
        fam = family_of(run)
        if fam is None:
            print(f"WARN: no family prefix matches {run}; skipped in all-runs figure")
            continue
        entries.append((FAMILY_ORDER.index(fam[0]), int(r["steps"]), run, r, fam[0]))
    if not entries:
        return
    entries.sort(key=lambda e: (e[0], e[1], e[2]))

    fig, axes = plt.subplots(1, 2, figsize=(15, 0.22 * len(entries) + 2.4), sharey=True)
    fig.suptitle(
        f"Physical-unit {derivative} — every run of the §9.2.13 sweep "
        "(dt = 1/30 s; log scale; 95% episode bootstrap CI; dashed = ground truth)",
        fontsize=12,
    )
    labels = [all_runs_label(run, r) for _, _, run, r, _ in entries]
    for ax, (m, gm, title) in zip(axes, DYNAMICS[derivative], strict=True):
        gt = float(next(iter(rows.values()))[gm])
        means, lows, highs, colors = [], [], [], []
        for _, _, _run, r, fam in entries:
            v = float(r[m])
            means.append(v)
            lows.append(v - float(r[f"{m}_lo"]))
            highs.append(float(r[f"{m}_hi"]) - v)
            colors.append(COLORS[fam])
        y = range(len(labels))
        ax.barh(
            y,
            means,
            xerr=[lows, highs],
            color=colors,
            alpha=0.85,
            error_kw={"lw": 1, "capsize": 2, "ecolor": "#333333"},
        )
        ax.set_xscale("log")
        ax.set_xlim(min(means) * 0.45, max(m + h for m, h in zip(means, highs, strict=True)) * 1.6)
        ax.axvline(gt, color="#000000", ls="--", lw=1.3)
        ax.text(gt * 1.06, -0.6, f"GT {gt:,.0f}", fontsize=8, color="#000000")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3, which="both")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_yticks(list(range(len(labels))), labels=[f"{label}  " for label in labels], fontsize=7)
    axes[0].invert_yaxis()
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = os.path.join(FIG_DIR, f"physical_{derivative}_all.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out} ({len(entries)} runs)")


def main() -> int:
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = load_rows()
    print(f"{len(rows)} runs with physical metrics")
    for derivative in DYNAMICS:
        fig_derivative_bars(rows, derivative)
        fig_derivative_all(rows, derivative)
        fig_derivative_budget(rows, derivative)
    fig_dynamics_budget(rows)
    fig_ratio_ladder(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

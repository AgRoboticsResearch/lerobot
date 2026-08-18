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
  figures/unified_h10_jitter.png — within-chunk jitter (rotational and XYZ)
    for EVERY run of the sweep (all 50 rows), 95% CIs, GT reference lines.
  figures/unified_h10_jitter_budget.png — jitter vs training steps: the
    two budget curves with CI bands plus single-budget reference stars,
    GT reference lines (mirrors unified_h10_budget.png for smoothness).

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

# Jitter-figure family resolution: prefix -> (family, short label). Order
# defines the bar grouping (long budget curves first, single-budget
# reference families after).
FAMILY_PREFIXES = [
    ("act_umi_identity_rot6d_1459_", "hist", "R18-VAE hist"),
    ("act_r50_v1_vae_", "r50v1", "R50-V1"),
    ("act_r18_l1_", "actl1", "ACT-L1"),
    ("act_r50_vae_", "r50vae", "R50-VAE"),
    ("act_r18_flow_u_lr1e5_", "flow", "ACT-flow"),
    ("pi05_port_openpi_recipe_", "port", "π0.5 port o-recipe"),
    ("pi05_lora_sroi_rot6d_h30_", "openpi", "openpi rot6d-h30"),
    ("pi05_lora_sroi_rot6d_", "openpi", "openpi rot6d"),
    ("pi05_lora_sroi_rotvec_", "openpi", "openpi rotvec"),
    ("pi05_port_", "port", "π0.5 port"),
    ("smolvla_rot6d_", "smol", "SmolVLA rot6d"),
    ("smolvla_axis_angle_", "smol", "SmolVLA axis-angle"),
]
FAMILY_ORDER = [fam for _, fam, _ in FAMILY_PREFIXES]


def fmt_step(step: int) -> str:
    if step % 1_000_000 == 0:
        return f"{step // 1_000_000}M"
    return f"{step // 1000}k"


def family_of(run: str) -> tuple[str, str] | None:
    for prefix, fam, label in FAMILY_PREFIXES:
        if run.startswith(prefix):
            return fam, label
    return None


def jitter_label(run: str, row: dict) -> str:
    _, base = family_of(run) or ("hist", run)
    seed = row.get("train_seed", "1000")
    s = f" s{seed}" if seed != "1000" else ""
    return f"{base}{s} {fmt_step(int(row['step']))}"


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
    port = sorted(
        (int(r["step"]), r) for run, r in rows.items()
        if run.startswith("pi05_port_seed1000_")
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
            (port, COLORS["port"], "π0.5 port (kiwi curve, 18 ckpts)", "^"),
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
    print(f"wrote {out} ({len(hist)} hist + {len(r50)} R50-V1 + {len(port)} port points)")


def fig_jitter(rows: dict[str, dict]) -> None:
    """Jitter for EVERY run of the sweep, grouped by family, log-x bars.

    Rotational jerk spans 0.027-0.76 deg and XYZ jerk 0.21-3.25 mm across
    the sweep, so both panels use a log x-axis; the dashed line marks the
    ground-truth jerk (identical across rows by the protocol invariance).
    """
    entries = []
    for run, r in rows.items():
        fam = family_of(run)
        if fam is None:
            print(f"WARN: no family prefix matches {run}; skipped in jitter figure")
            continue
        entries.append((FAMILY_ORDER.index(fam[0]), int(r["step"]), run, r, fam))
    if not entries:
        return
    entries.sort(key=lambda e: (e[0], e[1], e[2]))

    ref = next(iter(rows.values()))
    gt_rot = float(ref["gt_rot_jerk_deg"])
    gt_xyz = float(ref["gt_xyz_jerk_m"]) * 1000

    fig, axes = plt.subplots(1, 2, figsize=(15, 0.24 * len(entries) + 2.4), sharey=True)
    fig.suptitle(
        "Unified horizon-10 jitter — every run of the sweep "
        "(log scale; 95% episode bootstrap CI; dashed = ground truth)",
        fontsize=12,
    )
    labels = [jitter_label(run, r) for _, _, run, r, _ in entries]
    for ax, (met, title, scale, gt) in zip(
        axes,
        (
            ("rot_jerk_deg", "Within-chunk rotational jerk (deg)", 1, gt_rot),
            ("xyz_jerk_m", "Within-chunk XYZ jerk (mm)", 1000, gt_xyz),
        ),
    ):
        means, lows, highs, colors = [], [], [], []
        for _, _, run, r, (fam, _) in entries:
            v = float(r[met]) * scale
            means.append(v)
            lo, hi = r.get(f"{met}__ci_low"), r.get(f"{met}__ci_high")
            lows.append(v - float(lo) * scale if lo else 0.0)
            highs.append(float(hi) * scale - v if hi else 0.0)
            colors.append(COLORS[fam])
        y = range(len(labels))
        ax.barh(y, means, xerr=[lows, highs], color=colors, alpha=0.85,
                error_kw=dict(lw=1, capsize=2, ecolor="#333333"))
        ax.set_xscale("log")
        ax.set_xlim(min(means) * 0.45, max(m + h for m, h in zip(means, highs)) * 1.6)
        ax.axvline(gt, color="#000000", ls="--", lw=1.3)
        ax.text(gt * 1.06, -0.6, f"GT {gt:.3g}", fontsize=8, color="#000000")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3, which="both")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_yticks(list(range(len(labels))), labels=[f"{l}  " for l in labels], fontsize=7)
    axes[0].invert_yaxis()
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = os.path.join(FIG_DIR, "unified_h10_jitter.png")
    fig.savefig(out, dpi=200)
    print(f"wrote {out} ({len(entries)} runs)")


def fig_jitter_budget(rows: dict[str, dict]) -> None:
    """Jitter vs training steps — the budget-curve mirror of fig_budget."""
    hist = sorted(
        (int(r["step"]), r) for run, r in rows.items()
        if run.startswith("act_umi_identity_rot6d_1459_")
    )
    r50 = sorted(
        (int(r["step"]), r) for run, r in rows.items()
        if run.startswith("act_r50_v1_vae_seed1000_")
    )
    port = sorted(
        (int(r["step"]), r) for run, r in rows.items()
        if run.startswith("pi05_port_seed1000_")
    )
    ref = next(iter(rows.values()))
    gt_rot = float(ref["gt_rot_jerk_deg"])
    gt_xyz = float(ref["gt_xyz_jerk_m"]) * 1000

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Unified horizon-10 jitter vs training steps — smoothness budget curves",
        fontsize=12,
    )
    specs = [
        (axes[0], "rot_jerk_deg", "Within-chunk rotational jerk (deg)", 1, gt_rot),
        (axes[1], "xyz_jerk_m", "Within-chunk XYZ jerk (mm)", 1000, gt_xyz),
    ]
    for ax, met, ylab, scale, gt in specs:
        ymax = 0.0
        for series, color, label, marker in (
            (hist, COLORS["hist"], "ACT R18-VAE (historical, 30 ckpts)", "o"),
            (r50, COLORS["r50v1"], "ACT R50-V1 (fresh 1M run)", "s"),
            (port, COLORS["port"], "π0.5 port (kiwi curve, 18 ckpts)", "^"),
        ):
            if not series:
                continue
            xs = [s for s, _ in series]
            ys = [float(r[met]) * scale for _, r in series]
            ymax = max(ymax, max(ys))
            ax.plot(xs, ys, marker=marker, ms=4, lw=1.5, color=color, label=label)
            lo = series[0][1].get(f"{met}__ci_low")
            if lo:  # bootstrap band (deterministic across renders)
                ax.fill_between(
                    xs,
                    [float(r[f"{met}__ci_low"]) * scale for _, r in series],
                    [float(r[f"{met}__ci_high"]) * scale for _, r in series],
                    color=color, alpha=0.12, lw=0,
                )
        ax.axhline(gt, color="#000000", ls="--", lw=1.3, label=f"ground truth ({gt:.3g})")
        # single-budget references, stacked labels (jitter spreads widely, so
        # each star keeps its own vertical offset ladder)
        for i, (run, label, fam) in enumerate((
            ("pi05_lora_sroi_rot6d_seed1000_0020000steps", "openpi rot6d", "openpi"),
            ("pi05_lora_sroi_rotvec_seed1000_0020000steps", "openpi rotvec", "openpi"),
            ("pi05_lora_sroi_rot6d_h30_seed1000_0020000steps", "openpi h30", "openpi"),
            ("pi05_port_openpi_recipe_seed1000_020000steps", "π0.5 port o-recipe", "port"),
            ("act_r18_flow_u_lr1e5_seed2000_100000steps", "ACT-flow 50k", "flow"),
            ("act_r18_l1_seed2000_100000steps", "ACT-L1 100k", "actl1"),
            ("act_r50_vae_seed2000_100000steps", "R50-VAE 80k", "r50vae"),
            ("smolvla_rot6d_seed1000_100000steps", "SmolVLA rot6d", "smol"),
        )):
            if run not in rows:
                continue
            r = rows[run]
            x, y = int(r["step"]), float(r[met]) * scale
            ymax = max(ymax, y)
            ax.scatter([x], [y], color=COLORS[fam], marker="*", s=110, zorder=5)
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(4, 6 + i * 7), fontsize=7, color=COLORS[fam])
        ax.set_xscale("log")
        ax.set_xlabel("training steps")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, ymax * 1.15)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(FIG_DIR, "unified_h10_jitter_budget.png")
    fig.savefig(out, dpi=200)
    print(f"wrote {out} ({len(hist)} hist + {len(r50)} R50-V1 + {len(port)} port points)")


def main() -> int:
    rows = load_rows()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_metrics(rows)
    fig_budget(rows)
    fig_jitter(rows)
    fig_jitter_budget(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

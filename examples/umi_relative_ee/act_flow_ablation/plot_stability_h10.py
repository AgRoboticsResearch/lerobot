#!/usr/bin/env python3
"""Figures for the §9.2.19 cross-query prediction-stability evaluation.

Reads results_stability_h10/stability_h10.csv (compile_stability.py over the
kiwi stability tree) and renders:

  figures/stability_h10_scores.png — re-query disagreement at k=1 and k=10
    (XYZ mm + rotation deg, episode-balanced, 95% bootstrap CIs), all 17
    representative runs grouped by family.
  figures/stability_growth.png      — disagreement vs re-query interval
    (k = 1/5/10) with CI bands, family-colored.
  figures/stability_budget_grid.png — Fig. 15-style 2×3 budget view: rotation
    above XYZ, with columns for k = 1/5/10; curves for evaluated multi-budget
    families and stars for single-budget references.

Run via:
  /home/zfei/anaconda3/envs/py312/bin/python \
    examples/umi_relative_ee/act_flow_ablation/plot_stability_h10.py
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = "/mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/results_stability_h10/stability_h10.csv"
FIG_DIR = os.path.join(HERE, "figures")

# family -> (prefix, label, color); order defines the bar grouping.
FAMILY_PREFIXES = [
    ("act_umi_identity_rot6d_1459_", "hist", "ACT R18-VAE (historical)", "#444444"),
    ("act_r50_v1_vae_2frame_", "2frame", "ACT R50-VAE (ImageNet-V1) 2-frame (Q3)", "#8c564b"),
    ("act_r50_v1_vae_", "r50v1", "ACT R50-VAE (ImageNet-V1)", "#1f77b4"),
    ("act_r18_l1_", "actl1", "ACT-L1", "#2ca02c"),
    ("act_r50_vae_", "r50vae", "ACT R50-VAE (ImageNet-V2)", "#9467bd"),
    ("act_r18_flow_u_lr1e5_", "flow", "ACT-flow", "#d62728"),
    ("act_r18_diffusion_lr1e5_", "actdiff", "ACT-diffusion", "#ff7f0e"),
    ("diffusion_r18_", "dpr18", "Diffusion Policy r18", "#e377c2"),
    ("umi_official_dp_", "umidp", "released UMI DP", "#8c564b"),
    ("pi05_port_openpi_recipe_", "port20k", "π0.5 port o-recipe", "#7f7f7f"),
    ("pi05_port_", "port", "π0.5 port", "#ff7f0e"),
    ("smolvla_rot6d_", "smol", "SmolVLA rot6d", "#17becf"),
    ("smolvla_axis_angle_", "smolaa", "SmolVLA axis-ang", "#bcbd22"),
    ("smolvla_masked_", "smolmask", "SmolVLA masked", "#7f7f7f"),
]


def family_of(run: str):
    for prefix, _, label, color in FAMILY_PREFIXES:
        if run.startswith(prefix):
            return label, color
    return run, "#aaaaaa"


def short(run: str) -> str:
    seed = " s2000" if "_seed2000_" in run else ""
    raw_steps = int(run.rsplit("_", 1)[1].replace("steps", ""))
    steps = f"{raw_steps // 1_000_000}M" if raw_steps % 1_000_000 == 0 else f"{raw_steps // 1000}k"
    label = family_of(run)[0]
    return f"{label}{seed} {steps}"


def load() -> dict:
    rows: dict[str, dict[int, dict]] = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            run, k = r["run"], int(r["k"])
            rows.setdefault(run, {})[k] = r
    return rows


def fig_scores(rows: dict) -> None:
    order = sorted(rows, key=lambda r: (family_of(r)[0], float(rows[r][1]["xyz_overlap_mean_mm"])))
    labels = [short(r) for r in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 0.3 * len(order) + 2.0), sharey=True)
    fig.suptitle(
        "Cross-query prediction stability — disagreement between re-queried chunks "
        "(episode-balanced, 95% CI)",
        fontsize=12,
    )
    for ax, k, title in zip(
        axes,
        (1, 10),
        ("re-query interval k=1 (async-replan regime)", "k=10"),
        strict=True,
    ):
        means, lows, highs, colors = [], [], [], []
        for run in order:
            r = rows[run][k]
            v = float(r["xyz_overlap_mean_mm"])
            means.append(v)
            lows.append(v - float(r["xyz_overlap_mean_lo"]))
            highs.append(float(r["xyz_overlap_mean_hi"]) - v)
            colors.append(family_of(run)[1])
        y = range(len(order))
        ax.barh(
            y,
            means,
            xerr=[lows, highs],
            color=colors,
            alpha=0.85,
            error_kw={"lw": 1, "capsize": 2, "ecolor": "#333333"},
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("XYZ overlap disagreement (mm)")
        ax.grid(axis="x", alpha=0.3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_yticks(
        list(range(len(labels))), labels=[f"{label}  " for label in labels], fontsize=8
    )
    axes[0].invert_yaxis()
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(FIG_DIR, "stability_h10_scores.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out} ({len(order)} runs)")


def fig_growth(rows: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.suptitle("Re-query disagreement vs interval — every family drifts ~5x from k=1 to k=10; "
                 "SmolVLA starts 2x above the pack", fontsize=11)
    ks = [1, 5, 10]
    for run in sorted(rows):
        xyz = [float(rows[run][k]["xyz_overlap_mean_mm"]) for k in ks]
        rot = [float(rows[run][k]["rotation_overlap_mean_deg"]) for k in ks]
        label, color = family_of(run)
        lo = [float(rows[run][k]["xyz_overlap_mean_lo"]) for k in ks]
        hi = [float(rows[run][k]["xyz_overlap_mean_hi"]) for k in ks]
        axes[0].plot(ks, xyz, "-o", ms=4, lw=1.5, color=color, label=short(run))
        axes[0].fill_between(ks, lo, hi, color=color, alpha=0.08, lw=0)
        lo_r = [float(rows[run][k]["rotation_overlap_mean_lo"]) for k in ks]
        hi_r = [float(rows[run][k]["rotation_overlap_mean_hi"]) for k in ks]
        axes[1].plot(ks, rot, "-o", ms=4, lw=1.5, color=color)
        axes[1].fill_between(ks, lo_r, hi_r, color=color, alpha=0.08, lw=0)
    for ax, title in zip(
        axes, ("XYZ disagreement (mm)", "Rotation disagreement (deg)"), strict=True
    ):
        ax.set_xticks(ks)
        ax.set_xlabel("re-query interval k (frames, 30 fps)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(FIG_DIR, "stability_growth.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def _step_from_run(run: str) -> int:
    return int(run.rsplit("_", 1)[1].replace("steps", ""))


def fig_budget_grid(rows: dict) -> None:
    """Cross-query stability in the same compact 2×3 grammar as report Fig. 15."""
    curves = [
        (
            lambda run: run.startswith("act_r50_v1_vae_seed1000_"),
            "#1f77b4",
            "ACT R50-VAE (ImageNet-V1)",
            "s",
        ),
        (
            lambda run: run.startswith("pi05_port_seed1000_"),
            "#ff7f0e",
            "π0.5 port",
            "^",
        ),
        (
            lambda run: run.startswith("smolvla_rot6d_seed1000_"),
            "#17becf",
            "SmolVLA rot6d",
            "D",
        ),
    ]
    singles = [
        ("act_umi_identity_rot6d_1459_3000000steps", 3_000_000),
        ("act_r50_v1_vae_2frame_seed1000_0500000steps", 500_000),
        ("act_r50_v1_vae_noproprio_seed1000_0500000steps", 500_000),
        ("act_r18_l1_seed2000_100000steps", 100_000),
        ("act_r50_vae_seed2000_100000steps", 80_000),
        ("act_r18_flow_u_lr1e5_seed2000_100000steps", 50_000),
        ("act_r18_diffusion_lr1e5_seed1000_100000steps", 100_000),
        ("diffusion_r18_seed1000_100000steps", 100_000),
        ("umi_official_dp_seed1000_30000steps", 30_000),
        ("pi05_port_openpi_recipe_seed1000_020000steps", 20_000),
        ("smolvla_axis_angle_seed1000_100000steps", 100_000),
        ("smolvla_masked_seed1000_1000000steps", 1_000_000),
    ]
    metrics = (
        (
            "rotation_overlap_mean_deg",
            "rotation_overlap_mean_lo",
            "rotation_overlap_mean_hi",
            "Rotational overlap disagreement (deg)",
        ),
        (
            "xyz_overlap_mean_mm",
            "xyz_overlap_mean_lo",
            "xyz_overlap_mean_hi",
            "XYZ overlap disagreement (mm)",
        ),
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Cross-query prediction stability vs training budget — representative h10 sweep",
        fontsize=14,
    )
    for col, k in enumerate((1, 5, 10)):
        for row_index, (metric, lo_key, hi_key, ylabel) in enumerate(metrics):
            ax = axes[row_index, col]
            for select, color, label, marker in curves:
                selected = sorted(
                    (_step_from_run(run), per_k[k])
                    for run, per_k in rows.items()
                    if select(run)
                )
                if selected:
                    xs = [step for step, _ in selected]
                    ys = [float(result[metric]) for _, result in selected]
                    ax.plot(
                        xs,
                        ys,
                        marker=marker,
                        ms=5,
                        lw=1.7,
                        color=color,
                        label=label,
                    )
                    ax.fill_between(
                        xs,
                        [float(result[lo_key]) for _, result in selected],
                        [float(result[hi_key]) for _, result in selected],
                        color=color,
                        alpha=0.12,
                        lw=0,
                    )
            for run, step in singles:
                if run not in rows:
                    continue
                result = rows[run][k]
                value = float(result[metric])
                label, color = family_of(run)
                ax.errorbar(
                    [step],
                    [value],
                    yerr=[
                        [value - float(result[lo_key])],
                        [float(result[hi_key]) - value],
                    ],
                    fmt="*",
                    ms=9,
                    capsize=2,
                    lw=0.8,
                    color=color,
                    label=label,
                    zorder=5,
                )
            ax.axhline(0, color="k", ls="--", lw=1.1, label="perfect agreement (0)")
            ax.set_xscale("log")
            upper = ax.get_ylim()[1]
            ax.set_ylim(-0.04 * upper, upper)
            ax.set_xlabel("training steps")
            ax.set_ylabel(ylabel)
            ax.set_title(f"re-query interval k={k}")
            ax.grid(alpha=0.3)
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
    out = os.path.join(FIG_DIR, "stability_budget_grid.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    rows = load()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_scores(rows)
    fig_growth(rows)
    fig_budget_grid(rows)


if __name__ == "__main__":
    main()

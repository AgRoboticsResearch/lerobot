#!/usr/bin/env python3
"""Figures for the §9.2.15 salvage re-score (recovered seed-1000 matrix).

Reads results_salvage_h10/physical_jerk_h10.csv (compile_physical_jerk.py
over the kiwi salvage-eval tree; §9.2.9 protocol + §9.2.13 physical metrics)
plus the §9.2.13 CSV for the already-scored retrain rows, and renders:

  figures/salvage_h10_scores.png   — all 28 recovered runs: endpoint XYZ and
    true rot jerk (deg/s³) with the GT line, grouped by head type.
  figures/salvage_seed_trios.png   — training-seed trios at matched budget
    (endpoint), combining recovered seed-1000 rows with the §9.2.13
    seed-2000/3000 retrains; budget mismatches annotated.
  figures/salvage_budget.png       — 30k→100k budget pairs (seed 1000):
    endpoint and rot jerk.

Run via:
  /home/zfei/anaconda3/envs/py312/bin/python \
    examples/umi_relative_ee/act_flow_ablation/plot_salvage_h10.py
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SALVAGE_CSV = "/mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/results_salvage_h10/physical_jerk_h10.csv"
REF_CSV = "/mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/results_physical_jerk/physical_jerk_h10.csv"
FIG_DIR = os.path.join(HERE, "figures")

GT_ROT_JERK = 2793.0  # deg/s^3  (§9.2.13 GT reference)

# Head-type grouping for the 28 salvage rows.
GROUPS = [
    ("act_r18_vae", "ACT R18-VAE (deterministic)", "#1f77b4"),
    ("act_r34_vae", "ACT R34-VAE (deterministic)", "#1f77b4"),
    ("act_r50_vae", "ACT R50-VAE (deterministic)", "#1f77b4"),
    ("act_r50_large", "ACT R50-large (deterministic)", "#1f77b4"),
    ("act_r50_v1_vae", "ACT R50-V1 (deterministic)", "#1f77b4"),
    ("act_r18_l1", "ACT-L1 (deterministic)", "#2ca02c"),
    ("act_r18_flow", "ACT-flow (stochastic)", "#d62728"),
    ("act_r18_diffusion", "ACT-diffusion head (stochastic)", "#ff7f0e"),
    ("diffusion_r18", "Diffusion Policy r18 (stochastic)", "#9467bd"),
    ("umi_official", "released-UMI DP recipe (stochastic)", "#8c564b"),
]


def load(path: str) -> dict[str, dict]:
    with open(path) as f:
        return {r["run"]: r for r in csv.DictReader(f)}


def group_of(run: str) -> tuple[str, str]:
    for prefix, label, color in GROUPS:
        if run.startswith(prefix):
            return label, color
    raise KeyError(run)


def short(run: str) -> str:
    # act_r50_v1_vae_seed1000_100000steps -> R50-V1 s1000 100k
    fam = {
        "act_r18_vae": "R18-VAE", "act_r34_vae": "R34-VAE",
        "act_r50_vae": "R50-VAE", "act_r50_large": "R50L",
        "act_r50_v1_vae": "R50-V1", "act_r18_l1": "L1",
        "act_r18_flow_u_lr1e5": "flow-u 1e-5", "act_r18_flow_u_lr1e4": "flow-u 1e-4",
        "act_r18_flow_beta_lr1e4": "flow-β 1e-4", "act_r18_diffusion_lr1e5": "ACT-diff",
        "diffusion_r18": "DP-r18", "umi_official_dp": "UMI-DP",
        "umi_official_transformer_dp": "UMI-DP-T",
    }
    seed = run.split("_seed")[1][:4] if "_seed" in run else "1000"
    steps = run.rsplit("_", 1)[1].replace("steps", "")
    steps = f"{int(steps) // 1000}k"
    for k, v in fam.items():
        if run.startswith(k):
            return f"{v} s{seed} {steps}"
    return run


def fig_scores(rows: dict[str, dict]) -> None:
    order = sorted(rows, key=lambda r: (group_of(r)[0], float(rows[r]["xyz_end_mm"])))
    labels = [short(r) for r in order]
    endpoint = [float(rows[r]["xyz_end_mm"]) for r in order]
    jerk = [float(rows[r]["rot_jerk_deg_s3"]) for r in order]
    colors = [group_of(r)[1] for r in order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True)
    y = range(len(order))
    axes[0].barh(y, endpoint, color=colors)
    axes[0].set_xlabel("Endpoint XYZ error (mm, episode-balanced)")
    axes[0].set_title("Recovered runs — endpoint (t+10, unified protocol)")
    axes[0].axvline(9.0, color="gray", ls=":", lw=1)
    axes[0].axvline(11.0, color="gray", ls=":", lw=1)
    axes[0].text(11.05, 0.2, "§9.2.9 pack\n9–11 mm", fontsize=8, color="gray")
    axes[1].barh(y, jerk, color=colors)
    axes[1].axvline(GT_ROT_JERK, color="k", ls="--", lw=1.2)
    axes[1].text(GT_ROT_JERK * 1.03, 1.0, "GT 2793", fontsize=9)
    axes[1].set_xlabel("Rotational jerk (deg/s³, dt = 1/30 s)")
    axes[1].set_title("Recovered runs — true rotational jerk")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
    handles = [mpatches.Patch(color=c, label=l) for l, c in
               sorted({(g[0], g[1]) for g in (group_of(r) for r in order)})]
    axes[0].legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "salvage_h10_scores.png"), dpi=160)
    plt.close(fig)


def fig_trios(salvage: dict[str, dict], ref: dict[str, dict]) -> None:
    # (family label, [(run, csv, budget_note)])
    trios = [
        ("ACT-L1 @100k", [
            ("act_r18_l1_seed1000_100000steps", salvage, ""),
            ("act_r18_l1_seed2000_100000steps", ref, ""),
            ("act_r18_l1_seed3000_100000steps", ref, "")]),
        ("ACT R18-VAE @100k", [
            ("act_r18_vae_seed1000_100000steps", salvage, ""),
            ("act_r18_vae_seed2000_100000steps", salvage, ""),
            ("act_r18_vae_seed3000_100000steps", salvage, "")]),
        ("ACT-diff @100k", [
            ("act_r18_diffusion_lr1e5_seed1000_100000steps", salvage, ""),
            ("act_r18_diffusion_lr1e5_seed2000_100000steps", salvage, ""),
            ("act_r18_diffusion_lr1e5_seed3000_100000steps", salvage, "")]),
        ("DP-r18 @100k", [
            ("diffusion_r18_seed1000_100000steps", salvage, ""),
            ("diffusion_r18_seed2000_100000steps", salvage, ""),
            ("diffusion_r18_seed3000_100000steps", salvage, "30k")]),
        ("ACT-flow", [
            ("act_r18_flow_u_lr1e5_seed1000_100000steps", salvage, ""),
            ("act_r18_flow_u_lr1e5_seed2000_100000steps", ref, "50k"),
            ("act_r18_flow_u_lr1e5_seed3000_100000steps", ref, "50k")]),
        ("ACT R50-VAE", [
            ("act_r50_vae_seed1000_100000steps", salvage, ""),
            ("act_r50_vae_seed2000_100000steps", ref, "80k"),
            ("act_r50_vae_seed3000_100000steps", ref, "80k")]),
        ("ACT R50-V1", [
            ("act_r50_v1_vae_seed1000_100000steps", salvage, ""),
            ("act_r50_v1_vae_seed2000_100000steps", salvage, "70k"),
            ("act_r50_v1_vae_seed3000_100000steps", salvage, "20k†")]),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    xs = range(len(trios))
    for x, (label, entries) in zip(xs, trios):
        vals = [float(csv_[run]["xyz_end_mm"]) for run, csv_, _ in entries]
        for i, (run, csv_, note) in enumerate(entries):
            seed = run.split("_seed")[1][:4]
            full_budget = note == ""
            ax.scatter(x + (i - 1) * 0.18, float(csv_[run]["xyz_end_mm"]),
                       s=70 if full_budget else 45,
                       marker="o" if full_budget else "^",
                       color="C0" if seed == "1000" else ("C1" if seed == "2000" else "C2"),
                       zorder=3)
            if note:
                ax.annotate(note, (x + (i - 1) * 0.18, float(csv_[run]["xyz_end_mm"])),
                            textcoords="offset points", xytext=(0, -14),
                            fontsize=7, ha="center", color="gray")
        ax.plot([x - 0.28, x + 0.28], [sum(vals) / len(vals)] * 2,
                color="gray", lw=1, ls=":", zorder=2)
    ax.axhspan(9.0, 11.0, color="gray", alpha=0.12)
    ax.text(0.995, 10.0, "§9.2.9 pack 9–11 mm", fontsize=8, color="gray",
            ha="right", va="center", transform=ax.get_yaxis_transform())
    ax.set_xticks(list(xs))
    ax.set_xticklabels([t[0] for t in trios], fontsize=9)
    ax.set_ylabel("Endpoint XYZ error (mm, episode-balanced)")
    ax.set_title("Training-seed trios of the recovered matrix (dots = seeds; triangles = partial budget, † = torn 30k ckpt, scored @20k)")
    ax.grid(axis="y", alpha=0.3)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f"seed {s}")
               for c, s in (("C0", "1000"), ("C1", "2000"), ("C2", "3000"))]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "salvage_seed_trios.png"), dpi=160)
    plt.close(fig)


def fig_budget(rows: dict[str, dict]) -> None:
    pairs = [
        ("act_r18_flow_u_lr1e5", "flow-u 1e-5"),
        ("act_r18_l1", "ACT-L1"),
        ("act_r18_vae", "R18-VAE"),
        ("act_r50_v1_vae", "R50-V1"),
        ("act_r50_vae", "R50-VAE"),
        ("act_r18_diffusion_lr1e5", "ACT-diff"),
        ("diffusion_r18", "DP-r18"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for i, (prefix, label) in enumerate(pairs):
        lo = rows[f"{prefix}_seed1000_30000steps"]
        hi = rows[f"{prefix}_seed1000_100000steps"]
        for ax, key, unit in ((axes[0], "xyz_end_mm", "mm"),
                              (axes[1], "rot_jerk_deg_s3", "deg/s³")):
            vals = [float(lo[key]), float(hi[key])]
            ax.plot([0, 1], vals, "-o", color=f"C{i}", label=label if ax is axes[0] else None)
            ax.annotate(f"{vals[0]:.1f}", (0, vals[0]), textcoords="offset points",
                        xytext=(-4, 4), fontsize=7, ha="right")
            ax.annotate(f"{vals[1]:.1f}", (1, vals[1]), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, ha="left")
    axes[1].axhline(GT_ROT_JERK, color="k", ls="--", lw=1.2)
    axes[1].text(0.02, GT_ROT_JERK * 1.1, "GT 2793", fontsize=8)
    axes[1].set_yscale("log")
    for ax, title in ((axes[0], "Endpoint XYZ (mm)"), (axes[1], "Rot jerk (deg/s³, log)")):
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["30k", "100k"])
        ax.set_title(f"{title} — seed-1000 budget pairs")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "salvage_budget.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    salvage = load(SALVAGE_CSV)
    ref = load(REF_CSV)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_scores(salvage)
    fig_trios(salvage, ref)
    fig_budget(salvage)
    print("wrote salvage_h10_scores.png, salvage_seed_trios.png, salvage_budget.png")


if __name__ == "__main__":
    main()

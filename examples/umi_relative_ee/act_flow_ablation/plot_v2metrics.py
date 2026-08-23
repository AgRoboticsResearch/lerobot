#!/usr/bin/env python3
"""Render the v2-metric (L1 / per-dim MSE / accuracy@tau) figures from the
collector's v2 pass output.

Inputs  : <v2 root>/results/v2_run_summary.csv  (collect_results.py --v2_eval_roots ...)
Outputs : seed23k_v2metrics.png          (§9.2.6 — six seed-23k companions)
          historical_act_budget_curve.png (§9.2.7 — 30-point production-ACT curve)

Deterministic: fixed rcParams, no timestamps. Run with an ephemeral matplotlib
overlay so the shared training venv stays untouched:

  MPLCONFIGDIR=/tmp/lerobot-matplotlib uv run --with matplotlib python \
    examples/umi_relative_ee/act_flow_ablation/plot_v2metrics.py \
    --summary /mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/results/v2_run_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED23K_VARIANTS = ("act_r18_l1", "act_r50_vae", "act_r18_flow_u_lr1e5")
SEED23K_LABELS = {
    "act_r18_l1": "ACT-L1",
    "act_r50_vae": "ACT R50-VAE",
    "act_r18_flow_u_lr1e5": "ACT-flow (1e-5)",
}
HIST_VARIANT = "act_umi_identity_rot6d_1459"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "/mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/results/v2_run_summary.csv"
        ),
        help="v2_run_summary.csv from collect_results.py --v2_eval_roots.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "figures")
    return parser.parse_args()


def load_rows(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open() as file:
        return list(csv.DictReader(file))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def half_ci(row: dict[str, str], key: str) -> float:
    return (f(row, f"{key}_ci_high") - f(row, f"{key}_ci_low")) / 2


def plot_seed23k(rows: list[dict[str, str]], out_path: Path) -> None:
    panels = [
        ("xyz_l1_per_dim_m", "XYZ L1 / dim (mm)", 1e3),
        ("rotvec_l1_per_dim_deg", "Rotvec L1 / dim (°)", 1.0),
        ("action_acc_at_0p5", "accuracy@0.5", 1.0),
        ("action_acc_at_0p1", "accuracy@0.1", 1.0),
    ]
    seeds = sorted({int(row["training_seed"]) for row in rows})
    colors = {"act_r18_l1": "#4878d0", "act_r50_vae": "#ee854a", "act_r18_flow_u_lr1e5": "#6acc64"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for ax, (metric, ylabel, scale) in zip(axes.flat, panels):
        width = 0.8 / len(SEED23K_VARIANTS)
        for variant_index, variant in enumerate(SEED23K_VARIANTS):
            variant_rows = sorted(
                (row for row in rows if row["variant"] == variant),
                key=lambda row: int(row["training_seed"]),
            )
            positions = [
                seed_index + (variant_index - (len(SEED23K_VARIANTS) - 1) / 2) * width
                for seed_index in range(len(variant_rows))
            ]
            values = [f(row, metric) * scale for row in variant_rows]
            errors = [half_ci(row, metric) * scale for row in variant_rows]
            budget_note = {"act_r18_l1": "100k", "act_r50_vae": "80k", "act_r18_flow_u_lr1e5": "50k"}[variant]
            ax.bar(
                positions,
                values,
                width=width,
                yerr=errors,
                capsize=3,
                color=colors[variant],
                label=f"{SEED23K_LABELS[variant]} @{budget_note}",
            )
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([f"seed {seed}" for seed in seeds])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        if metric.startswith("action_acc"):
            ax.set_ylim(0.55, 1.0)
    axes.flat[0].set_title("Per-component L1 / accuracy@$\\tau$ — seed-23k companions (95% CI)")
    axes.flat[-1].legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_historical(rows: list[dict[str, str]], out_path: Path) -> None:
    hist = sorted(
        (row for row in rows if row["variant"] == HIST_VARIANT),
        key=lambda row: int(row["evaluated_step"]),
    )
    steps = [int(row["evaluated_step"]) for row in hist]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) XYZ endpoint with CI band
    ax = axes[0]
    xyz = [f(row, "xyz_end_m") * 1e3 for row in hist]
    low = [f(row, "xyz_end_m_ci_low") * 1e3 for row in hist]
    high = [f(row, "xyz_end_m_ci_high") * 1e3 for row in hist]
    ax.plot(steps, xyz, "o-", color="#4878d0", ms=3)
    ax.fill_between(steps, low, high, color="#4878d0", alpha=0.2, label="95% CI")
    ax.set_xscale("log")
    ax.set_xlabel("training steps")
    ax.set_ylabel("XYZ endpoint (mm)")
    ax.set_title("XYZ endpoint (95% CI band)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (b) accuracy@tau with capacity reference lines
    ax = axes[1]
    ax.plot(steps, [f(row, "action_acc_at_0p5") for row in hist], "s-", ms=3, label="acc@0.5")
    ax.plot(steps, [f(row, "action_acc_at_0p1") for row in hist], "o-", ms=3, label="acc@0.1")
    r50 = [row for row in rows if row["variant"] == "act_r50_vae"]
    l1 = [row for row in rows if row["variant"] == "act_r18_l1"]
    if r50:
        r50_acc = sum(f(row, "action_acc_at_0p1") for row in r50) / len(r50)
        ax.axhline(r50_acc, ls="--", color="#ee854a", label=f"R50-VAE@80k acc@0.1 ({r50_acc:.3f})")
    if l1:
        l1_acc = sum(f(row, "action_acc_at_0p1") for row in l1) / len(l1)
        ax.axhline(l1_acc, ls="--", color="#6acc64", label=f"ACT-L1@100k acc@0.1 ({l1_acc:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("training steps")
    ax.set_ylabel("accuracy@$\\tau$")
    ax.set_ylim(0.65, 1.0)
    ax.set_title("accuracy@$\\tau$ vs budget (R18-VAE)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")

    # (c) rotation 2nd-diff vs budget with GT line
    ax = axes[2]
    jerk = [f(row, "rot_jerk_deg") for row in hist]
    gt_jerk = f(hist[0], "gt_rot_jerk_deg")
    ax.plot(steps, jerk, "o-", ms=3, color="#d65f5f")
    ax.axhline(gt_jerk, color="gray", ls=":", label=f"ground truth ({gt_jerk:.3f}°)")
    best_index = min(range(len(jerk)), key=lambda i: jerk[i])
    ax.annotate(
        f"best @{steps[best_index] // 1000}k",
        xy=(steps[best_index], jerk[best_index]),
        xytext=(10, 18),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    ax.set_xscale("log")
    ax.set_xlabel("training steps")
    ax.set_ylabel("rotational 2nd-diff (°/step²)")
    ax.set_title("Within-chunk rotational 2nd-diff")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Historical production ACT (R18-VAE, seed 1000) — 30-point budget curve", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_r50_vs_r18(rows: list[dict[str, str]], out_path: Path) -> None:
    """§9.2.8: fresh R50-V1 1M curve vs historical R18 curve at horizon 30."""
    hist = sorted(
        (row for row in rows if row["variant"] == HIST_VARIANT),
        key=lambda row: int(row["evaluated_step"]),
    )
    r50 = sorted(
        (row for row in rows if row["variant"] == "act_r50_v1_vae_1m"),
        key=lambda row: int(row["evaluated_step"]),
    )
    if not r50:
        print("no act_r50_v1_vae_1m rows; skipping r50_vs_r18 figure")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) XYZ endpoint, both curves with CI bands
    ax = axes[0]
    for series, color, label in (
        (hist, "#4878d0", "R18-VAE (historical, 100k–3M)"),
        (r50, "#d65f5f", "R50-V1 (fresh, 100k–1M)"),
    ):
        steps = [int(row["evaluated_step"]) for row in series]
        ax.plot(steps, [f(row, "xyz_end_m") * 1e3 for row in series], "o-", ms=3, color=color, label=label)
        ax.fill_between(
            steps,
            [f(row, "xyz_end_m_ci_low") * 1e3 for row in series],
            [f(row, "xyz_end_m_ci_high") * 1e3 for row in series],
            color=color, alpha=0.15, lw=0,
        )
    ax.set_xscale("log")
    ax.set_xlabel("training steps")
    ax.set_ylabel("XYZ endpoint (mm)")
    ax.set_title("XYZ endpoint, horizon 30 (95% CI bands)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (b) accuracy@0.1
    ax = axes[1]
    ax.plot(
        [int(row["evaluated_step"]) for row in hist],
        [f(row, "action_acc_at_0p1") for row in hist],
        "o-", ms=3, color="#4878d0", label="R18-VAE",
    )
    ax.plot(
        [int(row["evaluated_step"]) for row in r50],
        [f(row, "action_acc_at_0p1") for row in r50],
        "s-", ms=3, color="#d65f5f", label="R50-V1",
    )
    ax.set_xscale("log")
    ax.set_xlabel("training steps")
    ax.set_ylabel("accuracy@0.1 (action)")
    ax.set_title("movement precision vs budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # (c) rotational 2nd-diff, both curves with GT line
    ax = axes[2]
    ax.plot(
        [int(row["evaluated_step"]) for row in hist],
        [f(row, "rot_jerk_deg") for row in hist],
        "o-", ms=3, color="#4878d0", label="R18-VAE",
    )
    ax.plot(
        [int(row["evaluated_step"]) for row in r50],
        [f(row, "rot_jerk_deg") for row in r50],
        "s-", ms=3, color="#d65f5f", label="R50-V1",
    )
    ax.axhline(f(hist[0], "gt_rot_jerk_deg"), color="gray", ls=":", label="ground truth")
    ax.set_xscale("log")
    ax.set_xlabel("training steps")
    ax.set_ylabel("rotational 2nd-diff (°/step²)")
    ax.set_title("within-chunk smoothness vs budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("ACT R50-V1 (fresh 1M run) vs historical R18-VAE — same protocol, horizon 30", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_seed23k(rows, args.out_dir / "seed23k_v2metrics.png")
    plot_historical(rows, args.out_dir / "historical_act_budget_curve.png")
    plot_r50_vs_r18(rows, args.out_dir / "r50_vs_r18_budget_curve.png")
    print(
        "wrote "
        + ", ".join(str(args.out_dir / n) for n in (
            "seed23k_v2metrics.png", "historical_act_budget_curve.png", "r50_vs_r18_budget_curve.png",
        ))
    )


if __name__ == "__main__":
    main()

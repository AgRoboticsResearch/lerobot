#!/usr/bin/env python
"""Render report figures from compact ablation CSV outputs."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["svg.hashsalt"] = "umi-act-flow-ablation"

DEFAULT_ROOT = Path("/media/zfei/Glowat512/projects/lerobot-arch-exp")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "figures"

LABELS = {
    "act_r18_vae": "ACT R18",
    "act_r34_vae": "ACT R34",
    "act_r50_vae": "ACT R50",
    "act_r50_large": "ACT R50 large",
    "act_r18_l1": "ACT-L1",
    "act_r18_flow_u_lr1e5": "ACT-flow 1e-5",
    "act_r18_flow_u_lr1e4": "ACT-flow 1e-4",
    "act_r18_flow_beta_lr1e4": "ACT-flow beta",
    "diffusion_r18": "DP R18",
    "umi_official_dp": "UMI ViT+U-Net",
    "umi_official_transformer_dp": "UMI ViT+Transformer",
}
ORDER = list(LABELS)
COLORS = {
    "act_r18_vae": "#4C78A8",
    "act_r34_vae": "#72A0C1",
    "act_r50_vae": "#1F5A94",
    "act_r50_large": "#163A5F",
    "act_r18_l1": "#59A14F",
    "act_r18_flow_u_lr1e5": "#E45756",
    "act_r18_flow_u_lr1e4": "#F28E8B",
    "act_r18_flow_beta_lr1e4": "#B33B3A",
    "diffusion_r18": "#F2CF5B",
    "umi_official_dp": "#B279A2",
    "umi_official_transformer_dp": "#7A5195",
}
EFFICIENCY_ANNOTATIONS = {
    "act_r18_vae": ((-5, -13), "right"),
    "act_r18_l1": ((5, 5), "left"),
    "act_r50_vae": ((-5, -14), "right"),
    "act_r50_large": ((5, 5), "left"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def ordered_variants(rows: list[dict[str, str]]) -> list[str]:
    present = {row["variant"] for row in rows}
    return [variant for variant in ORDER if variant in present]


def highest_budget_by_variant(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        variant = row["variant"]
        if variant not in selected or int(row["steps"]) > int(selected[variant]["steps"]):
            selected[variant] = row
    return selected


def budget_label(variant: str, steps: int) -> str:
    return f"{LABELS[variant]} ({steps // 1000}k)"


def seed_budget_label(variant: str, steps: int, num_training_seeds: int) -> str:
    label = budget_label(variant, steps)
    if num_training_seeds > 1:
        return f"{label[:-1]}, n={num_training_seeds})"
    return label


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{name}.svg"
    fig.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
    # Matplotlib emits trailing spaces in SVG path data; normalize generated
    # artifacts so repository whitespace checks remain useful and deterministic.
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fig.savefig(output_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(rows: list[dict[str, str]], output_dir: Path) -> None:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["variant"], int(row["steps"]))].append(row)
    panels = [
        ("ACT held-out L1", [v for v in ORDER if v.startswith("act_") and "flow" not in v], "l1_loss"),
        ("ACT-flow held-out velocity MSE", [v for v in ORDER if "flow" in v], "flow_loss"),
        (
            "Diffusion held-out noise MSE",
            ["diffusion_r18", "umi_official_dp", "umi_official_transformer_dp"],
            "loss",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for axis, (title, variants, metric) in zip(axes, panels, strict=True):
        for variant in variants:
            budgets = sorted(steps for candidate, steps in groups if candidate == variant)
            for steps in budgets:
                points = [row for row in groups[(variant, steps)] if row.get(metric)]
                if not points:
                    continue
                values_by_step: dict[int, list[float]] = defaultdict(list)
                for row in points:
                    values_by_step[int(row["step"])].append(float(row[metric]))
                validation_steps = sorted(values_by_step)
                means = np.asarray(
                    [float(np.mean(values_by_step[step])) for step in validation_steps]
                )
                standard_deviations = np.asarray(
                    [
                        float(np.std(values_by_step[step], ddof=1))
                        if len(values_by_step[step]) > 1
                        else np.nan
                        for step in validation_steps
                    ]
                )
                num_training_seeds = max(len(values) for values in values_by_step.values())
                axis.plot(
                    np.asarray(validation_steps) / 1000,
                    means,
                    marker="o",
                    linewidth=2,
                    markersize=4,
                    linestyle="-" if steps == max(budgets) else "--",
                    label=seed_budget_label(variant, steps, num_training_seeds),
                    color=COLORS[variant],
                )
                if np.isfinite(standard_deviations).any():
                    axis.fill_between(
                        np.asarray(validation_steps) / 1000,
                        means - standard_deviations,
                        means + standard_deviations,
                        where=np.isfinite(standard_deviations),
                        color=COLORS[variant],
                        alpha=0.14,
                        linewidth=0,
                    )
        axis.set_title(title)
        axis.set_xlabel("Optimizer steps (thousands)")
        axis.set_ylabel(metric.replace("_", " ").upper())
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=8, frameon=False)
    fig.suptitle("Validation learning curves (loss scales are not comparable across panels)", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "validation_learning_curves")


def plot_endpoint_bars(
    rows: list[dict[str, str]], seed_rows: list[dict[str, str]], output_dir: Path
) -> None:
    by_variant = highest_budget_by_variant(seed_rows)
    variants = ordered_variants(list(by_variant.values()))
    labels = [
        seed_budget_label(
            variant,
            int(by_variant[variant]["steps"]),
            int(by_variant[variant]["num_training_seeds"]),
        )
        for variant in variants
    ]
    run_lookup: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        run_lookup[(row["variant"], int(row["steps"]))].append(row)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    specifications = [
        ("xyz_end_m", "Endpoint translation error (mm)", 1000.0),
        ("rotation_end_deg", "Endpoint rotation error (deg)", 1.0),
    ]
    x = np.arange(len(variants))
    for axis, (metric, ylabel, scale) in zip(axes, specifications, strict=True):
        means = np.asarray([float(by_variant[v][f"{metric}_mean"]) * scale for v in variants])
        low = []
        high = []
        for variant in variants:
            aggregate = by_variant[variant]
            num_training_seeds = int(aggregate["num_training_seeds"])
            if num_training_seeds > 1:
                low.append(float(aggregate[f"{metric}_hierarchical_ci_low"]) * scale)
                high.append(float(aggregate[f"{metric}_hierarchical_ci_high"]) * scale)
            else:
                run = run_lookup[(variant, int(aggregate["steps"]))][0]
                low.append(float(run[f"{metric}_ci_low"]) * scale)
                high.append(float(run[f"{metric}_ci_high"]) * scale)
        low_array = np.asarray(low)
        high_array = np.asarray(high)
        axis.bar(
            x,
            means,
            color=[COLORS[variant] for variant in variants],
            yerr=np.vstack((means - low_array, high_array - means)),
            capsize=3,
            linewidth=0,
        )
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels, rotation=42, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Decoded endpoint error (95% episode or hierarchical training-seed bootstrap CI)"
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "decoded_endpoint_errors")


def plot_paired_improvements(
    rows: list[dict[str, str]], seed_rows: list[dict[str, str]], output_dir: Path
) -> None:
    desired_pairs = {
        ("act_r34_vae", "act_r18_vae"),
        ("act_r50_vae", "act_r18_vae"),
        ("act_r50_large", "act_r50_vae"),
        ("act_r18_flow_u_lr1e5", "act_r18_l1"),
        ("diffusion_r18", "act_r18_l1"),
        ("umi_official_dp", "diffusion_r18"),
        ("umi_official_transformer_dp", "umi_official_dp"),
    }
    metrics = ("xyz_end_m", "rotation_end_deg")
    candidates = [
        row
        for row in seed_rows
        if (row["variant"], row["baseline_variant"]) in desired_pairs and row["metric"] in metrics
    ]
    selected_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in candidates:
        key = (row["variant"], row["baseline_variant"], row["metric"])
        if key not in selected_by_key or int(row["steps"]) > int(selected_by_key[key]["steps"]):
            selected_by_key[key] = row
    selected = list(selected_by_key.values())
    pair_order = [
        pair for pair in desired_pairs if any((r["variant"], r["baseline_variant"]) == pair for r in selected)
    ]
    pair_order.sort(key=lambda pair: ORDER.index(pair[0]))
    if not pair_order:
        return
    lookup = {(row["variant"], row["baseline_variant"], row["metric"]): row for row in selected}
    y = np.arange(len(pair_order))
    fig, axis = plt.subplots(figsize=(10.5, max(4.2, 0.62 * len(pair_order))))
    offsets = {"xyz_end_m": -0.12, "rotation_end_deg": 0.12}
    metric_labels = {"xyz_end_m": "Translation endpoint", "rotation_end_deg": "Rotation endpoint"}
    metric_colors = {"xyz_end_m": "#2A9D8F", "rotation_end_deg": "#E76F51"}
    for metric in metrics:
        values = []
        lows = []
        highs = []
        indices = []
        for index, pair in enumerate(pair_order):
            row = lookup.get((*pair, metric))
            if row is None:
                continue
            value = float(row["paired_improvement_percent_mean"])
            num_training_seeds = int(row["num_training_seeds"])
            if num_training_seeds > 1:
                low = float(row["paired_improvement_percent_hierarchical_ci_low"])
                high = float(row["paired_improvement_percent_hierarchical_ci_high"])
            else:
                run = next(
                    candidate
                    for candidate in rows
                    if candidate["variant"] == row["variant"]
                    and candidate["baseline_variant"] == row["baseline_variant"]
                    and candidate["metric"] == metric
                    and candidate["steps"] == row["steps"]
                )
                low = float(run["paired_improvement_percent_ci_low"])
                high = float(run["paired_improvement_percent_ci_high"])
            values.append(value)
            lows.append(low)
            highs.append(high)
            indices.append(index)
        values_array = np.asarray(values)
        axis.errorbar(
            values_array,
            np.asarray(indices) + offsets[metric],
            xerr=np.vstack((values_array - np.asarray(lows), np.asarray(highs) - values_array)),
            fmt="o",
            capsize=3,
            color=metric_colors[metric],
            label=metric_labels[metric],
        )
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(
        y,
        [
            f"{LABELS[candidate]} vs {LABELS[baseline]} "
            f"({int(lookup[(candidate, baseline, metrics[0])]['steps']) // 1000}k, "
            f"n={lookup[(candidate, baseline, metrics[0])]['num_training_seeds']})"
            for candidate, baseline in pair_order
        ],
    )
    axis.set_xlabel("Paired error reduction (%) — positive favors candidate")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("Paired improvements with 95% episode/hierarchical bootstrap intervals")
    fig.tight_layout()
    save_figure(fig, output_dir, "paired_endpoint_improvements")


def plot_efficiency(
    summary_rows: list[dict[str, str]],
    seed_rows: list[dict[str, str]],
    run_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    latest_summaries = highest_budget_by_variant(seed_rows)
    summary_by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        summary_by_key[(row["variant"], int(row["steps"]))].append(row)
    run_by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in run_rows:
        if row.get("learnable_parameters") or row.get("parameters"):
            run_by_key[(row["variant"], int(row["steps"]))].append(row)
    points = [
        row
        for row in latest_summaries.values()
        if (row["variant"], int(row["steps"])) in summary_by_key
        and (row["variant"], int(row["steps"])) in run_by_key
    ]
    fig, axis = plt.subplots(figsize=(9.5, 6.0))
    for row in points:
        variant = row["variant"]
        steps = int(row["steps"])
        summaries = summary_by_key[(variant, steps)]
        runs = run_by_key[(variant, steps)]
        latency_ms = statistics.mean(float(summary["inference_median_seconds"]) for summary in summaries) * 1000
        parameters_m = statistics.mean(
            float(run.get("learnable_parameters") or run["parameters"]) for run in runs
        ) / 1e6
        endpoint_mm = float(row["xyz_end_m_mean"]) * 1000
        num_training_seeds = int(row["num_training_seeds"])
        if num_training_seeds > 1:
            low = float(row["xyz_end_m_hierarchical_ci_low"]) * 1000
            high = float(row["xyz_end_m_hierarchical_ci_high"]) * 1000
            axis.errorbar(
                latency_ms,
                endpoint_mm,
                yerr=np.asarray([[endpoint_mm - low], [high - endpoint_mm]]),
                fmt="none",
                capsize=3,
                color=COLORS[variant],
                alpha=0.8,
            )
        axis.scatter(
            latency_ms,
            endpoint_mm,
            s=35 + parameters_m * 1.2,
            color=COLORS[variant],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.7,
        )
        offset, horizontal_alignment = EFFICIENCY_ANNOTATIONS.get(variant, ((5, 4), "left"))
        axis.annotate(
            seed_budget_label(variant, steps, num_training_seeds),
            (latency_ms, endpoint_mm),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            ha=horizontal_alignment,
        )
    axis.set_xlabel("Median policy inference latency (ms)")
    axis.set_ylabel("Endpoint translation error (mm)")
    axis.set_title(
        "Accuracy–latency trade-off (area: online parameters; n>1 bars: hierarchical CI)"
    )
    axis.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir, "accuracy_latency_tradeoff")


def main() -> None:
    args = parse_args()
    result_dir = args.artifact_root / "results"
    validation_rows = read_csv(result_dir / "stage1_validation.csv")
    summary_rows = read_csv(result_dir / "stage1_variant_summary.csv")
    comparison_rows = read_csv(result_dir / "stage1_paired_comparisons.csv")
    seed_summary_rows = read_csv(result_dir / "training_seed_variant_summary.csv")
    seed_comparison_rows = read_csv(result_dir / "training_seed_paired_comparisons.csv")
    run_rows = read_csv(result_dir / "stage1_runs.csv")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    plot_learning_curves(validation_rows, args.output_dir)
    plot_endpoint_bars(summary_rows, seed_summary_rows, args.output_dir)
    plot_paired_improvements(comparison_rows, seed_comparison_rows, args.output_dir)
    plot_efficiency(summary_rows, seed_summary_rows, run_rows, args.output_dir)
    print(f"Rendered report figures under {args.output_dir}")


if __name__ == "__main__":
    main()

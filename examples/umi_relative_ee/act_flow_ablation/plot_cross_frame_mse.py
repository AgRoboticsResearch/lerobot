#!/usr/bin/env python3
"""Plot all-checkpoint adjacent-frame MSE budget curves at h10 and h30."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
COLORS = {
    "hist": "#444444",
    "r50v1": "#1f77b4",
    "2frame": "#1f9e89",
    "noproprio": "#8c564b",
    "port": "#ff7f0e",
    "smol": "#e377c2",
    "smolmask": "#7f7f7f",
    "lingbot": "#ff9896",
}
MARKERS = {
    "hist": "o",
    "r50v1": "s",
    "2frame": "x",
    "noproprio": "P",
    "port": "^",
    "smol": "D",
    "smolmask": "v",
    "lingbot": "*",
}
CURVE_LABELS = {
    "hist": "ACT R18-VAE historical (30 checkpoints)",
    "r50v1": "ACT R50-VAE ImageNet-V1 (10 checkpoints)",
    "2frame": "ACT R50-VAE ImageNet-V1, 2-frame (5 checkpoints)",
    "noproprio": "ACT R50-VAE ImageNet-V1, no proprio (5 checkpoints)",
    "port": "π0.5 port (19 checkpoints)",
    "smol": "SmolVLA full-width (10 checkpoints)",
    "smolmask": "SmolVLA masked (10 checkpoints)",
    "lingbot": "LingBot-VA (3 checkpoints; h10 only)",
}
EXPECTED_CURVE_COUNTS = {
    10: {
        "hist": 30,
        "r50v1": 10,
        "2frame": 5,
        "noproprio": 5,
        "port": 19,
        "smol": 10,
        "smolmask": 10,
        "lingbot": 3,
    },
    30: {
        "hist": 30,
        "r50v1": 10,
        "2frame": 5,
        "noproprio": 5,
        "port": 19,
        "smol": 10,
        "smolmask": 10,
        "lingbot": 0,
    },
}
PANELS = (
    ("action_cross_frame_mse_normalized", "Normalized action MSE"),
    ("xyz_cross_frame_mse_mm2_per_dim", "XYZ MSE (mm² per dim)"),
    ("rotation_geodesic_cross_frame_mse_deg2", "Rotation geodesic MSE (deg²)"),
)


def curve_family(run: str) -> str | None:
    if run.startswith("act_umi_identity_rot6d_1459_"):
        return "hist"
    if re.fullmatch(r"act_r50_v1_vae_2frame_1m_seed1000_\d{7}steps", run):
        return "2frame"
    if re.fullmatch(r"act_r50_v1_vae_noproprio_seed1000_\d{7}steps", run):
        return "noproprio"
    if re.fullmatch(r"act_r50_v1_vae_seed1000_\d{7}steps", run):
        return "r50v1"
    if re.fullmatch(r"pi05_port_seed1000_\d{7}steps", run):
        return "port"
    if re.fullmatch(r"smolvla_masked_seed1000_\d{7}steps", run):
        return "smolmask"
    if re.fullmatch(r"smolvla_rot6d_seed1000_\d{7}steps", run):
        return "smol"
    if re.fullmatch(r"lingbot_va_axis_angle_seed1000_\d+steps", run):
        return "lingbot"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(HERE / "repro" / "cross_frame_mse.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(HERE / "figures" / "cross_frame_mse_budget.png"),
    )
    args = parser.parse_args()
    with open(args.csv) as handle:
        rows = list(csv.DictReader(handle))

    by_horizon: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_horizon[int(row["horizon"])].append(row)
    training_steps = [int(row["step"]) for row in rows]
    x_limits = (min(training_steps) / 1.25, max(training_steps) * 1.15)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex="col")
    fig.suptitle(
        "Direct change between predictions queried at t and t+1 — full checkpoint inventory\n"
        "same-index whole-chunk MSE; independent inference draws; lower is more stable",
        fontsize=13,
    )
    for row_index, horizon in enumerate((10, 30)):
        horizon_rows = by_horizon[horizon]
        curves: dict[str, list[dict[str, str]]] = defaultdict(list)
        references = []
        for row in horizon_rows:
            family = curve_family(row["run"])
            if family is None:
                references.append(row)
            else:
                curves[family].append(row)
        expected_counts = EXPECTED_CURVE_COUNTS[horizon]
        observed_counts = {family: len(curves[family]) for family in expected_counts}
        if observed_counts != expected_counts:
            raise ValueError(
                f"h{horizon} budget-curve coverage mismatch: "
                f"{observed_counts} != {expected_counts}"
            )
        for column_index, (metric, title) in enumerate(PANELS):
            ax = axes[row_index, column_index]
            for family in CURVE_LABELS:
                series = sorted(curves[family], key=lambda item: int(item["step"]))
                if not series:
                    continue
                x = [int(item["step"]) for item in series]
                y = [float(item[metric]) for item in series]
                lo = [float(item[f"{metric}_lo"]) for item in series]
                hi = [float(item[f"{metric}_hi"]) for item in series]
                ax.plot(
                    x,
                    y,
                    marker=MARKERS[family],
                    ms=3.5,
                    lw=1.5,
                    color=COLORS[family],
                    label=CURVE_LABELS[family] if row_index == 0 and column_index == 0 else None,
                )
                ax.fill_between(x, lo, hi, color=COLORS[family], alpha=0.13, linewidth=0)
            if references:
                x = [int(item["step"]) for item in references]
                y = [float(item[metric]) for item in references]
                lo = [float(item[f"{metric}_lo"]) for item in references]
                hi = [float(item[f"{metric}_hi"]) for item in references]
                ax.errorbar(
                    x,
                    y,
                    yerr=[
                        [value - lower for value, lower in zip(y, lo, strict=True)],
                        [upper - value for value, upper in zip(y, hi, strict=True)],
                    ],
                    fmt="*",
                    ms=6,
                    color="#8c8c8c",
                    ecolor="#aaaaaa",
                    alpha=0.65,
                    elinewidth=0.7,
                    capsize=1,
                    label=(
                        "Single-budget / companion checkpoints"
                        if row_index == 0 and column_index == 0
                        else None
                    ),
                )
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlim(x_limits)
            ax.grid(alpha=0.25, which="both")
            ax.set_title(f"h{horizon}: {title}", fontsize=10)
            if row_index == 1:
                ax.set_xlabel("Training steps")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote {output} from {len(rows)} horizon rows")


if __name__ == "__main__":
    main()

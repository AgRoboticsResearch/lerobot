"""Plot one gap-free ASYNC ``latest_only`` episode in EE position and speed.

The default selection is the longest contiguous ``ik_ok`` segment in
``logs/async_20260807_150005``.  It produces a two-column, seven-row figure:
absolute EE target sent to the robot on the left and its first time derivative
on the right.  Yellow vertical lines mark the first executed tick of each new
``chunk_id``.

Usage::

    uv run python examples/umi_relative_ee/control_debug/make_async_latest_only_episode_zoom.py
"""
from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_STEM = REPO / "logs" / "async_20260807_150005"
DEFAULT_OUT = (
    REPO
    / "outputs"
    / "research_report"
    / "low_level_control_debug"
    / "sync_vs_latest_only"
    / "figures"
)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#66645f"
GRID = "#e1e0d9"
ARRIVAL = "#bdb006"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4", "#008300", "#eda100"]
LABELS = ["x", "y", "z", "rot_x", "rot_y", "rot_z", "gripper"]
COMMAND_UNITS = ["mm", "mm", "mm", "deg", "deg", "deg", "value"]
SPEED_UNITS = ["mm/s", "mm/s", "mm/s", "deg/s", "deg/s", "deg/s", "value/s"]
SCALE = np.array([1000.0, 1000.0, 1000.0, 180.0 / np.pi, 180.0 / np.pi,
                  180.0 / np.pi, 1.0])

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "legend.frameon": False,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", default=str(DEFAULT_STEM), help="Control-log stem")
    parser.add_argument("--segment", type=int, default=None,
                        help="Contiguous valid segment index; default selects the longest")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Figure output directory")
    return parser.parse_args()


def load(stem: Path) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    arrays = dict(np.load(stem.with_suffix(".npz"), allow_pickle=False).items())
    with stem.with_suffix(".csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return arrays, rows


def contiguous_segments(valid: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return []
    cuts = np.r_[0, np.flatnonzero(np.diff(indices) > 1) + 1, len(indices)]
    return [indices[a:b] for a, b in zip(cuts[:-1], cuts[1:], strict=True)]


def add_arrival_lines(axes: np.ndarray, arrival_times: np.ndarray) -> None:
    for ax in axes.flat:
        for i, arrival_time in enumerate(arrival_times):
            ax.axvline(
                arrival_time,
                color=ARRIVAL,
                lw=0.55,
                alpha=0.34,
                zorder=0,
                label="chunk arrival" if i == 0 and ax is axes.flat[0] else None,
            )


def summarize(
    stem: Path,
    d: dict[str, np.ndarray],
    rows: list[dict[str, str]],
    segment_index: int,
    segment: np.ndarray,
    arrival_rows: np.ndarray,
    speed: np.ndarray,
) -> dict:
    t = d["t_s"][segment]
    command = d["action_ee"][segment]
    dt = np.diff(t)
    return {
        "source": str(stem),
        "segment_index": segment_index,
        "start_log_row": int(segment[0]),
        "end_log_row": int(segment[-1]),
        "start_tick": int(d["tick"][segment[0]]),
        "end_tick": int(d["tick"][segment[-1]]),
        "start_time_s": float(t[0]),
        "end_time_s": float(t[-1]),
        "duration_s": float(t[-1] - t[0]),
        "executed_ticks": int(segment.size),
        "chunk_arrivals": int(arrival_rows.size),
        "chunk_ids": [int(x) for x in d["chunk_id"][arrival_rows]],
        "min_dt_ms": float(dt.min() * 1000.0),
        "max_dt_ms": float(dt.max() * 1000.0),
        "command_range_native": {
            label: [float(np.min(command[:, i])), float(np.max(command[:, i]))]
            for i, label in enumerate(LABELS)
        },
        "speed_p50": {
            label: float(np.percentile(np.abs(speed[:, i]), 50))
            for i, label in enumerate(LABELS)
        },
        "speed_p95": {
            label: float(np.percentile(np.abs(speed[:, i]), 95))
            for i, label in enumerate(LABELS)
        },
        "speed_abs_max": {
            label: float(np.max(np.abs(speed[:, i])))
            for i, label in enumerate(LABELS)
        },
        "rows_loaded": len(rows),
    }


def main() -> None:
    args = parse_args()
    stem = Path(args.stem).with_suffix("")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d, rows = load(stem)

    valid = d["ik_ok"].astype(bool)
    valid &= np.all(np.isfinite(d["action_ee"]), axis=1)
    segments = contiguous_segments(valid)
    if not segments:
        raise RuntimeError(f"No contiguous executed segment found in {stem}")
    if args.segment is None:
        segment_index = int(np.argmax([len(x) for x in segments]))
    else:
        segment_index = args.segment
    if segment_index < 0 or segment_index >= len(segments):
        raise IndexError(f"segment {segment_index} outside [0, {len(segments) - 1}]")
    segment = segments[segment_index]

    t = d["t_s"][segment]
    command = d["action_ee"][segment] * SCALE
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Selected segment has non-increasing timestamps")
    speed = np.diff(command, axis=0) / dt[:, None]
    speed_t = (t[1:] + t[:-1]) / 2.0

    chunk = d["chunk_id"][segment]
    arrival_mask = np.r_[True, chunk[1:] != chunk[:-1]]
    arrival_rows = segment[arrival_mask]
    arrival_times = d["t_s"][arrival_rows]

    fig, axes = plt.subplots(7, 2, figsize=(15, 18), sharex="col")
    add_arrival_lines(axes, arrival_times)
    for i, label in enumerate(LABELS):
        command_ax, speed_ax = axes[i]
        command_ax.plot(t, command[:, i], color=COLORS[i], lw=0.9, marker=".", ms=2.0,
                        label="action_ee sent")
        speed_ax.plot(speed_t, speed[:, i], color=COLORS[i], lw=0.8, marker=".", ms=1.7,
                      label="first derivative")
        speed_ax.axhline(0.0, color=MUTED, lw=0.65)
        command_ax.set_ylabel(f"{label}\n[{COMMAND_UNITS[i]}]", fontsize=8)
        speed_ax.set_ylabel(f"{label}\n[{SPEED_UNITS[i]}]", fontsize=8)
        command_ax.grid(True, alpha=0.22)
        speed_ax.grid(True, alpha=0.22)
        if i == 0:
            command_ax.set_title("Absolute EE target sent to robot", color=INK)
            speed_ax.set_title("Per-dimension first derivative", color=INK)
            command_ax.legend(fontsize=8, loc="upper left")
            speed_ax.legend(fontsize=8, loc="upper left")
        if i == 6:
            command_ax.set_xlabel("loop time [s]")
            speed_ax.set_xlabel("loop time [s]")

    axes[0, 0].text(
        0.995, 1.02,
        f"yellow = chunk arrival ({len(arrival_times)})",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=ARRIVAL,
    )
    fig.suptitle(
        "ASYNC latest_only — one gap-free executed segment\n"
        f"rows {segment[0]}–{segment[-1]}, ticks {int(d['tick'][segment[0]])}–{int(d['tick'][segment[-1]])}, "
        f"{len(segment)} ticks, {len(arrival_times)} chunk arrivals",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    figure_path = out / "async_latest_only_episode_ee_and_speed.png"
    fig.savefig(figure_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    summary = summarize(stem, d, rows, segment_index, segment, arrival_rows, speed)
    summary["figure"] = str(figure_path)
    summary_path = out / "async_latest_only_episode_ee_and_speed.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

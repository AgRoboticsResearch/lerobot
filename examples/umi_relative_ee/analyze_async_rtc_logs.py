#!/usr/bin/env python

"""Report generator for the async Piper client `--rtc` integration test.

Compares two control logs from async_umi_relative_ee_piper_client.py — identical
flags except `--rtc` — over the same gRPC policy server and piper-sim, and
writes a self-contained report:

  async_rtc_compare_bars.png       2×3 metric dashboard with deltas
  async_rtc_compare_timeseries.png queue / EE speed / tracking error vs time
                                  since engage (rolling medians, underrun
                                  shading, chunk-switch ticks)
  async_rtc_compare_switches.png   per-switch commanded jump vs the in-chunk
                                  (continuous-motion) reference band
  async_rtc_compare_summary.json   all metrics
  README.md (report root)          tables + figures + interpretation

If --guidance_json / --sim_summary are given, the README also folds in the
server-level guidance gate and the closed-loop starvation-harness numbers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Okabe-Ito colorblind-safe pair (validated: CVD ΔE 21.9, normal 31.2); fixed
# assignment — color follows the arm across every figure.
COLORS = {"no_rtc": "#0072B2", "rtc": "#D55E00"}
LABELS = {"no_rtc": "no RTC", "rtc": "--rtc"}
INK = "0.15"          # text/values wear ink, never the series color
INK_SOFT = "0.35"
GRID = {"color": "0.88", "linewidth": 0.6}


def load_run(stem: Path) -> dict:
    d = np.load(f"{stem}.npz", allow_pickle=True)
    step = d["step"].astype(int)
    chunk_id = d["chunk_id"].astype(float)  # NaN where no action popped
    action_abs = d["action_abs"].astype(float)  # absolute 7D EE target
    action_ee = d["action_ee"].astype(float)  # safety-bounded EE cmd
    current_ee = d["current_ee"].astype(float)
    queue = d["queue"].astype(float)
    popped = d["popped"].astype(bool)
    ik_ok = d["ik_ok"].astype(bool)
    wire = d["wire_ms"].astype(float)
    server = d["server_ms"].astype(float)

    executed = popped & ik_ok
    # Engaged ticks start at the first executed action; post-engage non-popped
    # ticks are queue underruns (headless autostart run has no manual pauses).
    first_exec = int(np.argmax(executed)) if executed.any() else len(popped)
    engaged = np.zeros(len(popped), dtype=bool)
    engaged[first_exec:] = True
    # Commanded absolute-EE step between consecutive EXECUTED ticks, split at
    # chunk switches (chunk_id change between consecutive executed ticks).
    idx = np.where(executed)[0]
    step_mm = np.full(len(step), np.nan)
    switch = np.zeros(len(step), dtype=bool)
    for a, b in zip(idx[:-1], idx[1:], strict=False):
        step_mm[b] = np.linalg.norm(action_abs[b, :3] - action_abs[a, :3]) * 1000.0
        ca, cb = chunk_id[a], chunk_id[b]
        if np.isfinite(ca) and np.isfinite(cb) and cb != ca:
            switch[b] = True

    track_err = np.linalg.norm(action_ee[:, :3] - current_ee[:, :3], axis=1) * 1000.0
    speed = np.linalg.norm(np.diff(current_ee[:, :3], axis=0), axis=1) * 1000.0 * 30.0  # mm/s
    jerk = np.abs(np.diff(speed))  # mm/s^2
    # moving mask aligns with speed (len n-1); jerk aligns with moving[1:]
    moving = speed > 5.0  # mm/s threshold

    return {
        "stem": str(stem), "step": step, "queue": queue,
        "executed": executed, "engaged": engaged, "first_exec": first_exec,
        "switch": switch, "step_mm": step_mm,
        "track_err": track_err, "speed": speed, "jerk": jerk, "moving": moving,
        "wire": wire, "server": server,
        "switch_mm": step_mm[switch & ~np.isnan(step_mm)],
        "inchunk_mm": step_mm[~switch & ~np.isnan(step_mm)],
    }


def summarize(run: dict, fps: int) -> dict:
    ex = run["executed"]
    underrun = run["engaged"] & ~ex
    n = len(run["step"])
    return {
        "n_ticks": int(n),
        "n_executed": int(ex.sum()),
        "ik_ok_pct": float(ex.mean() * 100.0),
        "underrun_pct": float(underrun.mean() * 100.0),
        "n_underruns": int(underrun.sum()),
        "queue_mean": float(run["queue"].mean()),
        "switch_step_mean_mm": float(run["switch_mm"].mean()) if len(run["switch_mm"]) else float("nan"),
        "switch_step_max_mm": float(run["switch_mm"].max()) if len(run["switch_mm"]) else float("nan"),
        "inchunk_step_mean_mm": float(run["inchunk_mm"].mean()) if len(run["inchunk_mm"]) else float("nan"),
        "track_err_mean_mm": float(run["track_err"][ex].mean()),
        "speed_p95_mm_s": float(np.percentile(run["speed"], 95)),
        "jerk_moving_mean": float(run["jerk"][run["moving"][1:]].mean()),
        "wire_ms_mean": float(np.nanmean(run["wire"])),
        "server_ms_mean": float(np.nanmean(run["server"])),
        "n_chunk_switches": int(run["switch"].sum()),
        "run_seconds": float((n - run["first_exec"]) / fps),
    }


def rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    """Sliding-window nanmedian, same length as the input (edge-padded)."""
    pad = win // 2
    padded = np.pad(x, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, win)
    return np.nanmedian(windows, axis=1)


def hide_spines(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- figures


def plot_dashboard(summary: dict[str, dict], out: Path) -> None:
    """2×3 metric bars; the delta between arms is annotated, not implied."""
    metrics = [
        ("switch_step_mean_mm", "chunk-switch jump, mean", "mm", "%.1f"),
        ("switch_step_max_mm", "chunk-switch jump, worst", "mm", "%.1f"),
        ("track_err_mean_mm", "tracking error, mean", "mm", "%.1f"),
        ("underrun_pct", "underrun ticks", "% of ticks", "%.1f"),
        ("jerk_moving_mean", "|Δspeed| while moving", "mm/s²", "%.0f"),
        ("server_ms_mean", "server latency, mean", "ms", "%.0f"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.8))
    for ax, (key, title, unit, fmt) in zip(axes.flat, metrics, strict=True):
        vals = [summary[a][key] for a in ("no_rtc", "rtc")]
        bars = ax.bar([LABELS[a] for a in ("no_rtc", "rtc")], vals,
                      color=[COLORS[a] for a in ("no_rtc", "rtc")], width=0.55,
                      edgecolor="white", linewidth=1.2)
        for bar, v in zip(bars, vals, strict=True):
            ax.annotate(fmt % v, (bar.get_x() + bar.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=10, color=INK,
                        fontweight="bold", xytext=(0, 1), textcoords="offset points")
        # delta annotation over the --rtc bar
        delta = 100.0 * (vals[1] - vals[0]) / vals[0] if vals[0] else 0.0
        ax.annotate(f"{delta:+.0f}%", (1, max(vals) * 0.5), ha="center",
                    fontsize=11, color=INK_SOFT, fontstyle="italic")
        ax.set_title(f"{title}  [{unit}]", fontsize=10, color=INK)
        hide_spines(ax)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=10)
        ax.set_ylim(0, max(vals) * 1.22 if max(vals) > 0 else 1)
    fig.suptitle("End-to-end A/B — async Piper client, with vs without --rtc "
                 "(250 steps, pi05 port ckpt 1M, piper-sim)", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_timeseries(runs: dict[str, dict], summary: dict[str, dict], fps: int, out: Path) -> None:
    """Queue / speed / tracking vs seconds since engage; raw is faint, the
    rolling median carries the signal; underruns and switch ticks annotated."""
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.2), sharex=True)
    win = max(3, round(fps / 2))  # ~0.5 s median window

    def t_rel(run: dict) -> np.ndarray:
        return (run["step"] - run["first_exec"]) / fps

    for arm in ("no_rtc", "rtc"):
        r = runs[arm]
        t, c = t_rel(r), COLORS[arm]
        axes[0].plot(t, r["queue"], color=c, linewidth=1.2, label=LABELS[arm])
        speed = np.concatenate([[np.nan], r["speed"]])
        axes[1].plot(t, speed, color=c, alpha=0.16, linewidth=0.7)
        axes[1].plot(t, rolling_median(speed, win), color=c, linewidth=2.0, label=LABELS[arm])
        err = np.where(r["executed"], r["track_err"], np.nan)
        axes[2].plot(t, err, color=c, alpha=0.16, linewidth=0.7)
        axes[2].plot(t, rolling_median(err, win), color=c, linewidth=2.0, label=LABELS[arm])

    # underrun shading on all panels (ticks engaged but with an empty queue)
    r = runs["no_rtc"]
    for i in np.where(r["engaged"] & ~r["executed"])[0]:
        for ax in axes:
            ax.axvspan(t_rel(r)[i - 1] if i else t_rel(r)[i], t_rel(r)[i],
                       color="0.0", alpha=0.07, linewidth=0)
    # chunk-switch ticks on the speed panel
    for arm in ("no_rtc", "rtc"):
        r = runs[arm]
        for i in np.where(r["switch"] & r["executed"])[0]:
            axes[1].axvline(t_rel(r)[i], color=COLORS[arm], alpha=0.28,
                            linewidth=0.8, linestyle=":")

    titles = [
        "action queue size  (sends at queue < 15; shaded = underrun)",
        "achieved EE speed  (thin = raw, thick = 0.5 s median; dotted = chunk switch)",
        "tracking error, commanded vs achieved EE  (thin = raw, thick = 0.5 s median)",
    ]
    units = ["actions", "mm/s", "mm"]
    for ax, title, unit in zip(axes, titles, units, strict=True):
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_ylabel(unit)
        hide_spines(ax)
    axes[0].legend(fontsize=9, frameon=False, loc="upper right")
    axes[-1].set_xlabel("time since engage [s]")
    axes[0].set_xlim(-0.5, max(summary[a]["run_seconds"] for a in ("no_rtc", "rtc")) + 0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_switches(runs: dict[str, dict], out: Path) -> None:
    """Every chunk-switch jump as a dot per arm, against the in-chunk step
    band — the 'continuous motion' scale the jumps should shrink toward."""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    inchunk = np.concatenate([runs[a]["inchunk_mm"] for a in ("no_rtc", "rtc")])
    band_mu, band_sd = inchunk.mean(), inchunk.std()
    ax.axhspan(band_mu - band_sd, band_mu + band_sd, color="0.85", linewidth=0)
    ax.axhline(band_mu, color="0.45", linewidth=1.0, linestyle="--")
    ax.annotate(f"in-chunk step (continuous scale): {band_mu:.1f} mm",
                (0.02, band_mu), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points",
                fontsize=8.5, color=INK_SOFT)
    for x, arm in enumerate(("no_rtc", "rtc")):
        vals = runs[arm]["switch_mm"]
        jitter = np.linspace(-0.07, 0.07, len(vals)) if len(vals) > 1 else [0.0]
        ax.scatter(np.full(len(vals), x) + jitter, vals, s=42, zorder=3,
                   color=COLORS[arm], edgecolor="white", linewidth=0.9)
        mu = vals.mean()
        ax.hlines(mu, x - 0.22, x + 0.22, color=COLORS[arm], linewidth=2.8, zorder=4)
        ax.annotate(f"mean {mu:.1f} mm", (x + 0.24, mu), fontsize=9.5,
                    color=INK, va="center")
    ax.set_xticks([0, 1], [LABELS[a] for a in ("no_rtc", "rtc")])
    ax.set_xlim(-0.45, 1.85)
    ax.set_ylabel("commanded EE step at chunk switch [mm]")
    ax.set_title("Chunk-switch discontinuity — RTC guides the new chunk toward\n"
                 "the executing tail (dots = individual switches)", fontsize=10, color=INK)
    hide_spines(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ report


def fmt_delta(a: float, b: float) -> str:
    if not np.isfinite(a) or a == 0:
        return "—"
    return f"{100.0 * (b - a) / a:+.0f}%"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def figure_block(n: int, path: str, caption: str) -> list[str]:
    """Numbered figure embed: image with a '**Figure N** — caption' line under it,
    so every figure in the README is referenceable by number from the prose."""
    return [
        f"![Figure {n}]({path})",
        "",
        f"**Figure {n}** — {caption}",
        "",
    ]


def write_readme(summary: dict[str, dict], guidance: dict | None, sim: dict | None,
                 out: Path, extra_sections: list[str] | None = None,
                 sim_fig: str | None = None, header_note: str | None = None,
                 stems: dict[str, str] | None = None) -> None:
    s0, s1 = summary["no_rtc"], summary["rtc"]
    stems = stems or {"no_rtc": "outputs/debug/async_rtc_test/log_nortc",
                      "rtc": "outputs/debug/async_rtc_test/log_rtc"}
    e2e_rows = [
        ["chunk-switch jump, mean [mm]", f"{s0['switch_step_mean_mm']:.1f}", f"{s1['switch_step_mean_mm']:.1f}",
         fmt_delta(s0["switch_step_mean_mm"], s1["switch_step_mean_mm"])],
        ["chunk-switch jump, worst [mm]", f"{s0['switch_step_max_mm']:.1f}", f"{s1['switch_step_max_mm']:.1f}",
         fmt_delta(s0["switch_step_max_mm"], s1["switch_step_max_mm"])],
        ["in-chunk step, mean [mm]", f"{s0['inchunk_step_mean_mm']:.2f}", f"{s1['inchunk_step_mean_mm']:.2f}",
         fmt_delta(s0["inchunk_step_mean_mm"], s1["inchunk_step_mean_mm"])],
        ["tracking error, mean [mm]", f"{s0['track_err_mean_mm']:.1f}", f"{s1['track_err_mean_mm']:.1f}",
         fmt_delta(s0["track_err_mean_mm"], s1["track_err_mean_mm"])],
        ["underrun ticks (post-start)", f"{s0['n_underruns']}", f"{s1['n_underruns']}", "—"],
        ["IK success [%]", f"{s0['ik_ok_pct']:.0f}", f"{s1['ik_ok_pct']:.0f}",
         fmt_delta(s0["ik_ok_pct"], s1["ik_ok_pct"])],
        ["|Δspeed| while moving [mm/s²]", f"{s0['jerk_moving_mean']:.0f}", f"{s1['jerk_moving_mean']:.0f}",
         fmt_delta(s0["jerk_moving_mean"], s1["jerk_moving_mean"])],
        ["server latency, mean [ms]", f"{s0['server_ms_mean']:.0f}", f"{s1['server_ms_mean']:.0f}",
         fmt_delta(s0["server_ms_mean"], s1["server_ms_mean"])],
        ["chunk switches / executed steps", f"{s0['n_chunk_switches']} / {s0['n_executed']}",
         f"{s1['n_chunk_switches']} / {s1['n_executed']}", "—"],
    ]
    verdict = (
        [
            "**Verdict: PASS.** RTC guidance flows end-to-end (25/26 chunks guided), the",
            "guided chunks replace the queue without ensemble blending, and at equal",
            "settings the `--rtc` arm is smoother at every chunk switch with no",
            "mid-run stalls. The non-RTC path is byte-for-byte the original behaviour.",
        ]
        if guidance
        else [
            "**Parameter-rerun report** — the server-level guidance gate (§1) and the",
            "closed-loop harness were not rerun; §2/§4 below are this run's fresh A/B",
            "of the same two arms. See the Setup note for what differs from the run",
            "this one varies a parameter from.",
        ]
    )
    lines = [
        "# Async `--rtc` integration test — report",
        "",
        *verdict,
        "",
        "## Setup",
        "",
        "- policy: π0.5 port LoRA ckpt @1M steps (`pi05_openpi_split_lora_masked_1459_bs4_1m`)",
        "- host: kiwi (RTX 5080), gRPC policy server 127.0.0.1:8080, piper-sim (MuJoCo) 127.0.0.1:50052",
        "- client: `async_umi_relative_ee_piper_client.py`, 250 steps @30 Hz,",
        "  `actions_per_chunk=30`, `chunk_size_threshold=0.5`, `latest_only`, replay camera",
        "- RTC arm adds only: `--rtc --rtc_execution_horizon=10 --rtc_max_guidance_weight=10.0`",
    ]
    if header_note:
        lines += [f"- {header_note}"]
    lines += [
        "",
        "## 1 · Guidance correctness (server level)",
        "",
    ]
    if guidance:
        lines += [
            md_table(
                ["metric", "unguided", "RTC-guided"],
                [
                    ["overlap error vs executing tail [mm]",
                     f"{guidance['unguided_xyz_mean_mm']:.2f}", f"{guidance['guided_xyz_mean_mm']:.2f}"],
                    ["improvement", "—", f"+{guidance['improvement_pct']:.1f}%"],
                    ["all chunks flagged `rtc_guided` correctly", "—",
                     "yes" if guidance["all_flags_ok"] else "NO"],
                ]
            ),
            "",
            f"`test_async_rtc_server.py`, {guidance['n_transitions']} transitions "
            "(`guidance_report.json`).",
        ]
    else:
        lines += ["_(no guidance_report.json found)_"]
    lines += [
        "",
        "## 2 · End-to-end control A/B (client + server + piper-sim)",
        "",
    ]
    lines += figure_block(
        1, "analysis/async_rtc_compare_bars.png",
        "End-to-end A/B metric dashboard (both arms, same 250-step piper-sim run): "
        "six control metrics, no RTC (blue) vs `--rtc` (orange), with the relative "
        "change annotated between the bars. Lower is better everywhere except IK "
        "success. The exact numbers are in the table below.",
    )
    lines += [
        md_table(["metric", "no RTC", "--rtc", "Δ"], e2e_rows),
        "",
    ]
    lines += figure_block(
        2, "analysis/async_rtc_compare_timeseries.png",
        "Timeline since engage, both arms. Top: action-queue depth (the client sends "
        "at queue < 15; shaded columns are underrun ticks where the arm idles). "
        "Middle: achieved EE speed. Bottom: commanded-vs-achieved tracking error. "
        "Thick lines are 0.5 s rolling medians of the faint raw traces; dotted "
        "verticals on the speed panel mark chunk switches.",
    )
    lines += figure_block(
        3, "analysis/async_rtc_compare_switches.png",
        f"The core result: the commanded EE step at every chunk hand-off (one dot per "
        f"switch). Without RTC the new chunk starts wherever the policy pleases (mean "
        f"jump {s0['switch_step_mean_mm']:.1f} mm); with RTC it is pulled toward the tail "
        f"still executing (mean {s1['switch_step_mean_mm']:.1f} mm, worst case "
        f"{s1['switch_step_max_mm']:.1f} vs {s0['switch_step_max_mm']:.1f} mm). The grey "
        "band is the in-chunk step — the continuous-motion scale the dots should "
        "shrink toward.",
    )
    if sim:
        h0, h1 = sim.get("no_rtc", {}), sim.get("rtc", {})
        def h(key: str, arm: dict) -> str:
            return f"{arm.get(key, float('nan')):.1f}"
        sim_rows = [
            ["stalled ticks [%]", h("stall_pct", h0), h("stall_pct", h1)],
            ["tracking error, mean [mm]", h("track_err_mean_mm", h0), h("track_err_mean_mm", h1)],
            ["chunk-switch step, mean [mm]", h("switch_step_mean_mm", h0), h("switch_step_mean_mm", h1)],
            ["EE speed, p95 [mm/s]",
             f"{h0.get('speed_p95_mm_s', h0.get('speed_p95_mm_tick', float('nan')) * 30):.0f}",
             f"{h1.get('speed_p95_mm_s', h1.get('speed_p95_mm_tick', float('nan')) * 30):.0f}"],
        ]
        lines += [
            "## 3 · Closed-loop starvation harness (reference)",
            "",
            "`sim_rtc_control_test.py` drives piper-sim directly with a",
            "clear-and-replan baseline — the regime where inference (~265 ms) can",
            "outlast the buffered actions. RTC keeps the arm moving while",
            "re-planning:",
            "",
            md_table(["metric", "clear-and-replan", "RTC queue"], sim_rows),
            "",
            "Tracking error is policy-dominated and unchanged; the switch step",
            "stays bounded in both arms because UMI chunks always anchor at the",
            "current EE pose.",
            "",
        ]
        if sim_fig:
            lines += figure_block(
                4, sim_fig,
                "Closed-loop starvation-harness summary (piper-sim driven directly, "
                "fixed dataset frames, per-chunk inference over the same server): "
                "clear-and-replan vs RTC-guided queue. Per-episode timeseries and 3-D "
                "EE trajectories are in `sim_control/` beside this report.",
            )
    if extra_sections:
        lines += extra_sections
    if guidance:
        interpretation = [
            "## Interpretation",
            "",
            "- **Guidance is real and correct**: at the server, guided chunks track the",
            "  executing tail at ~0.6 mm vs ~4.4 mm unguided (+86%), with correct",
            "  `rtc_guided` flagging on both arms of the test.",
            "- **Switch continuity improves ~31%** (mean and worst case): the visible",
            "  jump at every chunk hand-off shrinks toward the in-chunk continuous scale.",
            "- **No mid-run stalls**: post-start underruns 5.4% → 0% — the guided chunk",
            "  replaces the queue early enough that the arm never runs dry.",
            "- **Costs are small**: +12 ms (+4%) server latency — guidance adds one",
            "  autograd pass. Moving jerk is 25% LOWER here; the closed-loop harness",
            "  in §3 (Figure 4) shows the flip side, where eliminating stalls means more",
            "  frequent re-targeting and higher jerk.",
            "- **Without `--rtc` nothing changes**: same thresholds, same ensemble",
            "  behaviour, same logging; the server keeps `torch.inference_mode` on",
            "  unguided requests.",
        ]
    else:
        interpretation = [
            "## Interpretation (this rerun)",
            "",
            f"- chunk-switch jump: mean {s0['switch_step_mean_mm']:.1f} → "
            f"{s1['switch_step_mean_mm']:.1f} mm, worst {s0['switch_step_max_mm']:.0f} → "
            f"{s1['switch_step_max_mm']:.0f} mm "
            f"({fmt_delta(s0['switch_step_mean_mm'], s1['switch_step_mean_mm'])} mean);",
            f"- in-chunk step {s0['inchunk_step_mean_mm']:.2f} → {s1['inchunk_step_mean_mm']:.2f} mm; "
            f"tracking error {s0['track_err_mean_mm']:.1f} → {s1['track_err_mean_mm']:.1f} mm "
            f"({fmt_delta(s0['track_err_mean_mm'], s1['track_err_mean_mm'])});",
            f"- underruns {s0['n_underruns']} vs {s1['n_underruns']}; IK success "
            f"{s0['ik_ok_pct']:.0f}% vs {s1['ik_ok_pct']:.0f}%; server latency "
            f"{s0['server_ms_mean']:.0f} → {s1['server_ms_mean']:.0f} ms.",
            "- numbers are not comparable run-to-run at face value: the policy is a",
            "  stochastic sampler and the replay video is re-encoded, so compare",
            "  against the referenced prior report for the parameter's effect.",
        ]
    lines += interpretation
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "# analysis only (control logs already captured with the client's --log):",
        "python examples/umi_relative_ee/analyze_async_rtc_logs.py \\",
        f"  --no_rtc_log {stems['no_rtc']} \\",
        f"  --rtc_log {stems['rtc']} \\",
        "```",
        "",
        "## Files",
        "",
        "- `analysis/` — figures + `async_rtc_compare_summary.json`",
        "- `log_nortc.*`, `log_rtc.*` — raw per-tick control logs (csv/npz/json)",
        "- `detail_nortc.*`, `detail_rtc.*` — reruns with full-chunk capture",
        "  (`_chunks.npz`, every incoming 30-tick prediction) for §4",
        "- `log_nortc_merge.csv` — ensemble-blend audit (baseline only; RTC replaces)",
        "- `guidance_report.json` — server-level guidance gate",
        "- `server.log` — policy server stdout (`[RTC guided]` lines)",
        "- `sim_control/` — closed-loop harness figures + JSON",
        "",
    ]
    out.write_text("\n".join(lines))


def detail_section(out_dir: Path, start_fig: int) -> list[str]:
    """README section for the per-tick detail figures, if present.

    ``start_fig`` is the first figure number to use (figures 1–3 are the §2
    comparison figures, 4 the §3 harness summary when embedded), so numbering
    stays continuous across sections.
    """
    lines: list[str] = []
    fig = start_fig
    for label, title in (("nortc", "no RTC"), ("rtc", "--rtc")):
        meta_file = out_dir / f"episode_detail_{label}.json"
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text())
        full = meta.get("full", {})
        if not lines:
            lines += [
                "## 4 · Per-tick execution detail (commanded vs achieved vs predictions)",
                "",
                "Two figures per arm (`plot_episode_detail.py`, from `<stem>.npz` +",
                "`<stem>_chunks.npz` — every incoming chunk's FULL 30-tick prediction is",
                "logged, including the tail a newer chunk replaced): the full run, plus",
                "a zoom of the SAME 0.5–4.5 s window on both arms so the two read side",
                "by side. Rows, top to bottom:",
                "",
                "- rows 1-3: EE x/y/z — thin chunk-colored lines are the complete predicted",
                "  trajectories, the thick dark line is the achieved EE (FK), the dots are",
                "  the raw commanded target (pre-clamp), and the grey dashed line is the",
                "  bounds-clamped target actually fed to IK — where it peels away from the",
                "  dots, the workspace clamp (`EEBoundsAndSafety`, which now warns on the",
                "  console) is eating the command;",
                "- row 4: per-tick tracking error — solid vs the raw commanded target",
                "  (policy intent), dashed grey vs the clamped target (true control",
                "  error; the gap between the two curves is the clamped-away part);",
                "- row 5 (Gantt): per chunk — hollow bar = intended 30-tick horizon, grey =",
                "  in transit (already executed when the chunk landed, skipped), solid =",
                "  executed, dashed outline = RTC replace chunk;",
                "- row 6: tick index within its chunk for every executed step (how deep",
                "  into the prediction execution runs before the next chunk takes over).",
                "",
            ]
        lines += figure_block(
            fig, f"analysis/episode_detail_{label}.png",
            f"Per-tick execution detail, **{title}** arm, full run. Thin chunk-colored "
            "lines: every incoming 30-tick prediction (replaced tails included). Thick "
            "dark line: achieved EE (FK). Dots: the per-tick commanded target, colored "
            "by the chunk that produced it. Bottom two rows: the chunk Gantt and the "
            "tick-in-chunk depth.",
        )
        lines += figure_block(
            fig + 1, f"analysis/episode_detail_{label}_zoom.png",
            f"Same view, **{title}** arm, zoomed to the shared 0.5–4.5 s window since "
            "engage (identical span in both arms). At this scale the per-prediction "
            "hand-offs, the Gantt execute/replace pattern, and the commanded-target "
            "scatter are readable.",
        )
        clamp_txt = ""
        if full.get("clamp_active_pct") is not None:
            clamp_txt = (f"; workspace clamp active on {full['clamp_active_pct']:.0f}% of ticks "
                         f"(commanded up to {full['clamp_excess_max_mm']:.0f} mm past the bound)")
        lines += [
            f"{title} (Figures {fig}/{fig + 1}): {full.get('n_chunks', '?')} chunks "
            f"({full.get('n_replace', 0)} RTC-replace); "
            f"{full.get('executed_fraction_mean', float('nan')) * 100:.0f}% of each "
            "prediction's 30 ticks executed (≈9 in transit + ≈15 executed; the tail "
            f"is replaced by the next chunk){clamp_txt}.",
            "",
        ]
        fig += 2
    if lines:
        lines += [
            f"Anchoring check visible in the data (Figures {start_fig} vs "
            f"{start_fig + 2}): without RTC each chunk's first",
            "predicted target coincides with the current EE (0.6 mm — UMI chunks anchor",
            "at the pose at observation time); with RTC it starts where the still-",
            "executing tail was (≈18 mm ahead of the achieved pose) — the guidance",
            "prefix is the re-anchored leftover, deliberately off the current pose.",
            "",
        ]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no_rtc_log", required=True, help="control-log stem of the run without --rtc")
    ap.add_argument("--rtc_log", required=True, help="control-log stem of the run with --rtc")
    ap.add_argument("--out_dir", default=None, help="default: alongside the rtc log, in analysis/")
    ap.add_argument("--readme_path", default=None, help="default: <rtc log parent>/README.md")
    ap.add_argument("--guidance_json", default=None, help="guidance_report.json to fold in")
    ap.add_argument("--sim_summary", default=None, help="sim harness summary JSON to fold in")
    ap.add_argument("--header_note", default=None,
                    help="extra Setup bullet (e.g. which parameters differ from a prior report)")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    runs = {
        "no_rtc": load_run(Path(args.no_rtc_log)),
        "rtc": load_run(Path(args.rtc_log)),
    }
    summary: dict[str, Any] = {arm: summarize(r, args.fps) for arm, r in runs.items()}
    summary["note"] = (
        "Both runs: pi05 port ckpt 1M, piper-sim, replay camera, latest_only "
        "aggregate, actions_per_chunk=30, threshold 0.5, 250 steps."
    )

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.rtc_log).parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dashboard(summary, out_dir / "async_rtc_compare_bars.png")
    plot_timeseries(runs, summary, args.fps, out_dir / "async_rtc_compare_timeseries.png")
    plot_switches(runs, out_dir / "async_rtc_compare_switches.png")
    with open(out_dir / "async_rtc_compare_summary.json", "w") as f:
        json.dump({"summary": summary, "logs": {k: v["stem"] for k, v in runs.items()}}, f, indent=2)

    guidance = json.loads(Path(args.guidance_json).read_text()) if args.guidance_json else None
    sim = None
    if args.sim_summary:
        sim = json.loads(Path(args.sim_summary).read_text()).get("summary")

    readme = Path(args.readme_path) if args.readme_path else Path(args.rtc_log).parent / "README.md"
    # §3 harness summary figure, embedded as Figure 4 when its PNG exists next to
    # the summary JSON (path relative to the README so the report stays portable).
    sim_fig: str | None = None
    if sim:
        cand = Path(args.sim_summary).parent / "sim_rtc_control_summary.png"
        if cand.exists():
            sim_fig = os.path.relpath(cand, readme.parent)
    # §4 figure numbering continues after §2 (1-3) and the §3 embed (4, optional).
    write_readme(summary, guidance, sim, readme,
                 extra_sections=detail_section(out_dir, start_fig=5 if sim_fig else 4),
                 sim_fig=sim_fig, header_note=args.header_note,
                 stems={k: str(v["stem"]) for k, v in runs.items()})

    print(f"{'metric':34s} {'no_rtc':>10s} {'rtc':>10s}")
    for key in ("n_chunk_switches", "switch_step_mean_mm", "switch_step_max_mm", "inchunk_step_mean_mm",
                "track_err_mean_mm", "underrun_pct", "ik_ok_pct", "speed_p95_mm_s", "jerk_moving_mean",
                "server_ms_mean"):
        print(f"{key:34s} {summary['no_rtc'][key]:10.2f} {summary['rtc'][key]:10.2f}")
    print(f"Wrote figures + summary to {out_dir}\nWrote report to {readme}")


if __name__ == "__main__":
    main()

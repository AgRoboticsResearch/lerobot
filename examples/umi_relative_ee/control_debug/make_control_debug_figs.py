"""Generate low-level control-debug figures + computed stats from two --log runs.

Reads a SYNC + ASYNC control-log triplet (.npz/.csv/.json) written by the Piper
deploy scripts' `--log` mode (see examples/umi_relative_ee/control_logger.py) and
writes, into an output dir under outputs/research_report/:

  <out>/figures/*.png   one figure per analysis view (embedded in the .md report)
  <out>/computed_stats.json   headline numbers the markdown cites (in sync w/ figs)

Both runs share ONE schema (same NPZ arrays / JSON meta keys), so sync vs async
panels are directly diffable side by side.

Usage:
  uv run python examples/umi_relative_ee/control_debug/make_control_debug_figs.py \
      [sync_stem] [async_stem] [out_dir]

  sync_stem / async_stem : path or stem of a control log under logs/ (or absolute),
                           e.g. sync_20260807_091050  (defaults: the runs below)
  out_dir                : where figures/ + computed_stats.json go
                           (default: outputs/research_report/low_level_control_debug)

Paths are resolved relative to the repo root (the parent three levels up from this
file), so the script can be run from anywhere.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # examples/umi_relative_ee/control_debug -> repo root
LOGS = REPO / "logs"
OUTDIR_DEFAULT = REPO / "outputs" / "research_report" / "low_level_control_debug"

DEFAULT_SYNC = "sync_20260807_091050"
DEFAULT_ASYNC = "async_20260807_091447"

# ── Validated default palette (dataviz skill / palette.md), light surface ──
SURF = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
PAUSED_FILL = "#e9e7e0"  # faint surface tint for PAUSED spans

plt.rcParams.update({
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "figure.dpi": 110,
    "legend.frameon": False,
    "legend.fontsize": 8,
})

EE_LABELS = ["x [m]", "y [m]", "z [m]", "rot_x [rad]", "rot_y [rad]", "rot_z [rad]", "gripper"]
JOINT_LABELS = [f"joint{i+1} [deg]" for i in range(6)]  # Piper reports degrees
FPS_TARGET = 30.0
TARGET_DT = 1000.0 / FPS_TARGET


def resolve_stem(arg: str) -> Path:
    """Resolve a user-supplied stem to a control-log stem Path (no suffix)."""
    p = Path(arg)
    if not p.is_absolute():
        p = LOGS / arg
    if p.is_dir():
        # Directory containing the triplet: pick the unique .npz stem.
        npzs = list(p.glob("*.npz"))
        if len(npzs) != 1:
            raise SystemExit(f"{p}: expected exactly one .npz, found {len(npzs)}")
        p = npzs[0]
    if p.suffix in (".csv", ".npz", ".json"):
        p = p.with_suffix("")
    if not p.with_suffix(".npz").exists():
        raise SystemExit(f"control log not found: {p}.npz")
    return p


def load(stem: Path) -> dict:
    npz = np.load(stem.with_suffix(".npz"))
    meta = json.loads(stem.with_suffix(".json").read_text())
    d = {k: npz[k] for k in npz.files}
    # The NPZ holds only numeric fields; the string columns (state, skip_reason)
    # live in the CSV. Read them and align by row order (identical to the NPZ).
    with stem.with_suffix(".csv").open() as fh:
        rows = list(csv.DictReader(fh))
    d["state"] = np.array([r["state"] for r in rows])
    d["skip_reason"] = np.array([r["skip_reason"] for r in rows])
    return {
        "d": d,
        "meta": meta,
        "tag": stem.name.split("_")[0],  # 'sync' | 'async'
        "summary": meta.get("summary", {}),
        "args": meta.get("args", {}),
    }


def pct(a: np.ndarray, q: float) -> float:
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


def mean(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else float("nan")


def shade_paused(ax, t: np.ndarray, state: np.ndarray) -> None:
    """Faintly shade PAUSED spans so INFERENCE gaps read as white.

    NOTE: axvspan extends the x-axis to the full timeline, so this MUST NOT be
    used on valid-only figures (it would defeat the zoom to the executed window).
    """
    paused = state == "PAUSED"
    if not paused.any():
        return
    lo = None
    for i in range(len(t)):
        if paused[i] and lo is None:
            lo = t[i]
        if (not paused[i] or i == len(t) - 1) and lo is not None:
            hi = t[i]
            ax.axvspan(lo, hi, color=PAUSED_FILL, alpha=0.6, lw=0, zorder=0)
            lo = None


def xlim_valid(ax, t: np.ndarray, valid: np.ndarray, pad_frac: float = 0.03) -> None:
    """Zoom the x-axis to the executed (valid) time window (with a small pad)."""
    tv = t[valid]
    if tv.size == 0:
        return
    lo, hi = float(tv.min()), float(tv.max())
    pad = max((hi - lo) * pad_frac, 1e-3)
    ax.set_xlim(lo - pad, hi + pad)


# ────────────────────────── Figure 1: EE pose command (sent to IK) ──────
# VALID-only: a tick is executed iff ik_ok (popped → IK solved → written). While
# PAUSED the robot accepts NO policy output, so action_ee is plotted for executed
# ticks only — same valid-only treatment as the joint command (Fig 2).
def fig_ee_pose(runs: list[dict], figdir: Path) -> None:
    fig, axes = plt.subplots(7, 2, figsize=(11, 12.5), sharex="col")
    for c, run in enumerate(runs):
        d = run["d"]
        t = d["t_s"]
        valid = d["ik_ok"].astype(bool)
        axes[0, c].set_title(run["title"], loc="center", color=INK)
        for r in range(7):
            ax = axes[r, c]
            ax.plot(t[valid], d["action_ee"][valid, r], color=CAT[r], lw=1.0,
                    marker="o", markersize=2.5, markeredgecolor=SURF, markeredgewidth=0.3)
            ax.set_ylabel(EE_LABELS[r], fontsize=8)
            if r == 0:
                ax.plot([], [], color=CAT[r], lw=1.4, label="executed action_ee (sent to IK)")
                ax.legend(loc="upper right", fontsize=7)
            ax.tick_params(labelsize=7)
        xlim_valid(axes[0, c], t, valid)
        axes[-1, c].set_xlabel("loop time  t [s]  (zoomed to executed window)")
    fig.suptitle("EE pose command sent to IK — executed ticks only (action_ee)",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(figdir / "01_ee_pose_command.png", bbox_inches="tight")
    plt.close(fig)


# ─────────────────── Figure 2: joint command vs read-back ───────────────
def fig_joints(runs: list[dict], figdir: Path) -> None:
    fig, axes = plt.subplots(6, 2, figsize=(11, 11.5), sharex="col")
    for c, run in enumerate(runs):
        d = run["d"]
        t = d["t_s"]
        valid = d["ik_ok"].astype(bool)
        axes[0, c].set_title(run["title"], loc="center", color=INK)
        for r in range(6):
            ax = axes[r, c]
            ax.plot(t[valid], d["ik_joints_rad"][valid, r], color=BLUE, lw=0.7,
                    marker="o", markersize=2.5, markeredgecolor=SURF, markeredgewidth=0.3,
                    label="IK joint cmd → robot")
            ax.plot(t[valid], d["current_joints_rad"][valid, r], color=ORANGE, lw=0.6, alpha=0.9,
                    marker="o", markersize=2, markeredgecolor=SURF, markeredgewidth=0.25,
                    label="read-back (pre-cmd) joints")
            ax.set_ylabel(JOINT_LABELS[r], fontsize=8)
            ax.tick_params(labelsize=7)
            if r == 0 and c == 0:
                ax.legend(loc="upper right")
        xlim_valid(axes[0, c], t, valid)
        axes[-1, c].set_xlabel("loop time  t [s]  (zoomed to executed window)")
    fig.suptitle("Joint command vs read-back — executed ticks only (6 joints)", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(figdir / "02_joint_command_vs_readback.png", bbox_inches="tight")
    plt.close(fig)


# ─────────────────── Figure 3: joint motion decomposition ──────────────
def _joint_signals(d: dict) -> dict:
    """Three executed-tick joint signals [deg], max over 6 joints:
    tracking = |q_cmd-q_meas|, command = |Δq_cmd|, measured = |Δq_meas|."""
    ok = d["ik_ok"].astype(bool)
    if not ok.any():
        return {k: np.array([np.nan]) for k in ("track", "cmd", "meas")}
    qc = d["ik_joints_rad"][ok].astype(float)   # degrees (mislabeled "_rad")
    qm = d["current_joints_rad"][ok].astype(float)
    return {
        "track": np.max(np.abs(qc - qm), axis=1),
        "cmd": np.max(np.abs(np.diff(qc, axis=0)), axis=1),
        "meas": np.max(np.abs(np.diff(qm, axis=0)), axis=1),
    }


def fig_joint_decomposition(runs: list[dict], figdir: Path) -> None:
    """Separate the three signals the tracking-gap metric conflates.

    ``joint_delta_max_rad`` is |q_cmd(t)-q_meas(t)| = a TRACKING GAP, not per-tick
    motion. Left: that gap vs the actual command increment and measured motion. Right:
    per-joint tracking gap. Values are DEGREES (the "_rad" joint columns are mislabeled —
    Piper reports degrees; an earlier draft np.rad2deg'd them and inflated ~57x).
    """
    s_run, a_run = runs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    labels = ["Δq_tracking\n|cmd−meas|", "Δq_command\n|Δcmd|", "Δq_measured\n|Δmeas|"]
    x = np.arange(3)
    w = 0.38
    sg = _joint_signals(s_run["d"])
    ag = _joint_signals(a_run["d"])
    s_means = [float(np.mean(sg["track"])), float(np.mean(sg["cmd"])), float(np.mean(sg["meas"]))]
    a_means = [float(np.mean(ag["track"])), float(np.mean(ag["cmd"])), float(np.mean(ag["meas"]))]
    ax1.bar(x - w / 2, s_means, w, color=BLUE, label=f"SYNC (n={int(s_run['d']['ik_ok'].sum())})")
    ax1.bar(x + w / 2, a_means, w, color=ORANGE, label=f"ASYNC (n={int(a_run['d']['ik_ok'].sum())})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("mean per-tick joint signal [deg]")
    ax1.set_title("Tracking gap vs command step vs measured motion", color=INK2)
    ax1.legend()
    for xi, v in zip(x - w / 2, s_means):
        ax1.text(xi, v, f"{v:.2f}°", ha="center", va="bottom", fontsize=7, color=INK2)
    for xi, v in zip(x + w / 2, a_means):
        ax1.text(xi, v, f"{v:.2f}°", ha="center", va="bottom", fontsize=7, color=INK2)

    def per_joint(run: dict) -> np.ndarray:
        d = run["d"]
        ok = d["ik_ok"].astype(bool)
        return (np.abs(d["ik_joints_rad"][ok] - d["current_joints_rad"][ok]).mean(0)
                if ok.any() else np.full(6, np.nan))

    jx = np.arange(6)
    w2 = 0.38
    ax2.bar(jx - w2 / 2, per_joint(s_run), w2, color=BLUE)
    ax2.bar(jx + w2 / 2, per_joint(a_run), w2, color=ORANGE)
    ax2.set_xticks(jx)
    ax2.set_xticklabels(["j1", "j2", "j3", "j4", "j5", "j6"])
    ax2.set_ylabel("mean tracking gap [deg]")
    ax2.set_title("Per-joint tracking gap", color=INK2)

    fig.suptitle("Joint motion decomposition (degrees) — the “52°/tick” was a unit bug; "
                 "true motion is <1°/tick", fontsize=10.5, y=1.0)
    fig.tight_layout()
    fig.savefig(figdir / "03_joint_decomposition.png", bbox_inches="tight")
    plt.close(fig)


# ───────────────────────── Figure 4: loop timing ────────────────────────
def fig_timing(runs: list[dict], figdir: Path) -> None:
    a_run = runs[1]
    fig = plt.figure(figsize=(11, 9.5))
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.22)

    # 3a: tick_dt over time, per run — linear 0–100 ms (the ~3 s spike clips off the top)
    for c, run in enumerate(runs):
        ax = fig.add_subplot(gs[0, c])
        d = run["d"]
        t = d["t_s"]
        ax.plot(t, d["tick_dt_ms"], color=BLUE, lw=1.0, label="tick_dt (full period)")
        ax.plot(t, d["work_ms"], color=ORANGE, lw=0.8, alpha=0.8, label="work (pre-sleep)")
        ax.axhline(TARGET_DT, color=GREEN, lw=1.0, ls="--", label=f"target {TARGET_DT:.1f} ms (30 Hz)")
        ax.axhline(1000.0 / 20, color=MUTED, lw=0.8, ls=":", label="20 ms")
        ax.set_ylim(0, 100)
        ax.set_title(run["title"], color=INK)
        ax.set_ylabel("ms")
        if c == 0:
            ax.legend(loc="upper right", ncol=2, fontsize=7)
        ax.set_xlabel("loop time [s]")
    fig.suptitle("Loop timing — full-tick period vs work-only time", fontsize=12, y=0.98)

    # 3b: steady-state tick_dt histograms (exclude warmup = first 2 s)
    for c, run in enumerate(runs):
        ax = fig.add_subplot(gs[1, c])
        d = run["d"]
        steady = d["tick_dt_ms"][(d["t_s"] > 2.0) & np.isfinite(d["tick_dt_ms"])]
        ax.hist(steady, bins=60, color=BLUE, alpha=0.85)
        ax.axvline(TARGET_DT, color=GREEN, lw=1.0, ls="--")
        ax.axvline(mean(steady), color=ORANGE, lw=1.0, ls="-", label=f"mean {mean(steady):.1f} ms")
        ax.set_title(f"{run['tag'].upper()} steady-state tick_dt (t>2 s, n={steady.size})", color=INK2)
        ax.set_xlabel("tick_dt [ms]")
        ax.set_ylabel("ticks")
        ax.legend()

    # 3c: async network latency (e2e / wire / server) — sync has none
    ax = fig.add_subplot(gs[2, 0])
    d = a_run["d"]
    m = np.isfinite(d["e2e_ms"])
    ax.hist(d["e2e_ms"][m], bins=50, color=VIOLET, alpha=0.85, label=f"e2e (n={int(m.sum())})")
    ax.axvline(mean(d["e2e_ms"]), color=INK, lw=1.0, ls="-", label=f"mean {mean(d['e2e_ms']):.1f} ms")
    ax.set_title("ASYNC request→first-action latency  e2e_ms", color=INK2)
    ax.set_xlabel("ms")
    ax.set_ylabel("chunks")
    ax.legend()

    # 3d: per-tick work-vs-sleep breakdown (async, steady) — jitter source
    ax = fig.add_subplot(gs[2, 1])
    d = a_run["d"]
    m = (d["t_s"] > 2.0) & np.isfinite(d["work_ms"])
    ax.scatter(d["work_ms"][m], d["tick_dt_ms"][m], s=6, color=AQUA, alpha=0.5)
    lim = np.percentile(d["work_ms"][m], 99)
    ax.set_xlim(0, max(lim * 1.1, 40))
    ax.set_ylim(0, 80)
    ax.axhline(TARGET_DT, color=GREEN, lw=1.0, ls="--", label="33.3 ms")
    ax.set_title("ASYNC work_ms vs tick_dt_ms (steady)", color=INK2)
    ax.set_xlabel("work_ms")
    ax.set_ylabel("tick_dt_ms")
    ax.legend()

    fig.savefig(figdir / "04_loop_timing.png", bbox_inches="tight")
    plt.close(fig)


# ─────────────────── Figure 5: stutter + queue (valid-only) ─────────────
def fig_stutter(runs: list[dict], figdir: Path) -> None:
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.22)

    # 5a: ee_delta_m over time (executed only)
    for c, run in enumerate(runs):
        ax = fig.add_subplot(gs[0, c])
        d = run["d"]
        t = d["t_s"]
        valid = d["ik_ok"].astype(bool)
        ax.plot(t[valid], d["ee_delta_m"][valid], color=MAGENTA, lw=1.1)
        ax.axhline(run["args"].get("max_ee_step_m", 0.05), color=RED, lw=1.0, ls="--",
                   label=f"max_ee_step {run['args'].get('max_ee_step_m', 0.05)} m")
        ax.set_title(f"{run['tag'].upper()} per-step EE move  ee_delta_m", color=INK2)
        ax.set_ylabel("m")
        ax.legend(loc="upper right")
        xlim_valid(ax, t, valid)
    fig.suptitle("Stutter metrics — executed ticks only", fontsize=12, y=0.98)

    # 5b: joint_delta_max_rad over time (executed only)
    for c, run in enumerate(runs):
        ax = fig.add_subplot(gs[1, c])
        d = run["d"]
        t = d["t_s"]
        valid = d["ik_ok"].astype(bool)
        ax.plot(t[valid], d["joint_delta_max_rad"][valid], color=RED, lw=1.1)
        ax.axhline(5, color=MUTED, lw=0.8, ls=":", label="5°")
        ax.axhline(15, color=RED, lw=0.9, ls="--", label="15°")
        ax.set_title(f"{run['tag'].upper()} max joint tracking gap (degrees)", color=INK2)
        ax.set_ylabel("deg")
        ax.legend(loc="upper right")
        xlim_valid(ax, t, valid)

    # 5c: queue depth at executed ticks
    for c, run in enumerate(runs):
        ax = fig.add_subplot(gs[2, c])
        d = run["d"]
        t = d["t_s"]
        valid = d["ik_ok"].astype(bool)
        ax.plot(t[valid], d["queue"][valid], color=VIOLET, lw=1.0)
        ax.set_title(f"{run['tag'].upper()} queue depth at executed ticks", color=INK2)
        ax.set_ylabel("queued actions")
        ax.set_xlabel("loop time [s]  (zoomed to executed window)")
        xlim_valid(ax, t, valid)

    fig.savefig(figdir / "05_stutter_and_queue.png", bbox_inches="tight")
    plt.close(fig)


# ─────────────────── Figure 6: execution strategy ───────────────────────
def _ee_step_metrics(d: dict) -> dict:
    """Per-tick EE-target step on executed ticks + chunk-switch flags + reversal."""
    valid = d["ik_ok"].astype(bool)
    idx = np.where(valid)[0]
    ee = d["action_ee"][idx, :3]
    q = d["queue"][idx]
    steps = np.diff(ee, axis=0)                  # (M,3): step arriving at each next executed tick
    mag = np.linalg.norm(steps, axis=1) * 1e3    # mm
    mdir = steps.mean(0)
    n = float(np.linalg.norm(mdir))
    mdir = mdir / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    signed = steps @ mdir * 1e3                  # mm along the principal motion direction (for the plot)
    # Jitter metric: how often two adjacent steps oppose (dot<0) — the back-and-forth signal.
    if steps.shape[0] > 1:
        reversal = float(np.mean(np.sum(steps[1:] * steps[:-1], axis=1) < 0))
    else:
        reversal = float("nan")
    is_switch = q[1:] == 29                      # this executed tick opened a fresh chunk (SYNC replay)
    return {
        "t": d["t_s"][idx][1:], "mag": mag, "signed": signed, "is_switch": is_switch,
        "reversal": reversal,
        "mag_switch": mag[is_switch] if is_switch.any() else np.array([]),
    }


def fig_execution_strategy(runs: list[dict], figdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    for c, run in enumerate(runs):
        m = _ee_step_metrics(run["d"])
        tt, mag, signed, sw = m["t"], m["mag"], m["signed"], m["is_switch"]

        ax0 = axes[0, c]
        ax0.scatter(tt[~sw], mag[~sw], s=10, color=BLUE, alpha=0.7, label="within-chunk step")
        if sw.any():
            ax0.scatter(tt[sw], mag[sw], s=55, color=RED, marker="x", zorder=5,
                        label=f"chunk-switch (n={int(sw.sum())})")
        ax0.set_title(f"{run['tag'].upper()} per-tick EE step magnitude", color=INK2)
        ax0.set_ylabel("EE step [mm]")
        ax0.legend(fontsize=7)

        ax1 = axes[1, c]
        pos, neg = signed >= 0, signed < 0
        ax1.scatter(tt[pos], signed[pos], s=10, color=BLUE, alpha=0.7, label="+ along motion")
        ax1.scatter(tt[neg], signed[neg], s=12, color=RED, alpha=0.85, label="− (reversal)")
        ax1.axhline(0, color=MUTED, lw=0.8)
        ax1.set_title(f"signed EE step along motion dir  —  reversal {m['reversal']:.0%}", color=INK2)
        ax1.set_ylabel("signed step [mm]")
        ax1.set_xlabel("loop time [s]")
        ax1.legend(fontsize=7)
    fig.suptitle("Execution strategy — SYNC chunk replay (switch spikes, low reversal) "
                 "vs ASYNC temporal ensemble (continuous re-blend, high reversal)",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figdir / "06_execution_strategy.png", bbox_inches="tight")
    plt.close(fig)


# ─────────────────────── computed stats for the .md ─────────────────────
def block(run: dict) -> dict:
    d = run["d"]
    s = run["summary"]
    t = d["t_s"]
    warm = t < 2.0
    steady_dt = d["tick_dt_ms"][~warm & np.isfinite(d["tick_dt_ms"])]
    idx = int(np.nanargmax(d["tick_dt_ms"]))
    # NOTE on units: ik_joints_rad / current_joints_rad / joint_delta_max_rad are in
    # DEGREES despite the "_rad" suffix (piper.read_joints reports degrees; values match
    # START_POSE_DEG). Do NOT np.rad2deg them — earlier figures did and inflated ~57x.
    # A tick is valid (executed) iff ik_ok: popped + IK solved + written to motors.
    valid = d["ik_ok"].astype(bool)
    paused = d["state"] == "PAUSED"
    m = valid.astype(int)
    n_seg = int(((m[1:] == 1) & (m[:-1] == 0)).sum()) + (1 if m[0] == 1 else 0)
    per_joint = (np.abs(d["ik_joints_rad"][valid] - d["current_joints_rad"][valid]).mean(0)
                 if valid.any() else np.full(6, np.nan))
    # Three joint signals [deg], max over the 6 joints, on executed ticks:
    #   tracking = |q_cmd(t) - q_meas(t)|    (what joint_delta_max_rad actually is)
    #   command  = |q_cmd(t) - q_cmd(t-1)|   (per-tick command increment)
    #   measured = |q_meas(t) - q_meas(t-1)| (actual per-tick robot motion)
    if valid.any():
        qc = d["ik_joints_rad"][valid].astype(float)
        qm = d["current_joints_rad"][valid].astype(float)
        j_track = np.max(np.abs(qc - qm), axis=1)
        j_cmd = np.max(np.abs(np.diff(qc, axis=0)), axis=1)
        j_meas = np.max(np.abs(np.diff(qm, axis=0)), axis=1)
    else:
        j_track = j_cmd = j_meas = np.array([np.nan])
    exec_ee_step = (np.linalg.norm(np.diff(d["action_ee"][valid, :3], axis=0), axis=1)
                    if valid.sum() > 1 else np.array([np.nan]))
    # Phantom spikes: action_ee popped while PAUSED (SYNC logs these; never executed).
    paused_ee = d["action_ee"][paused, :3]
    paused_step = (np.linalg.norm(np.diff(paused_ee, axis=0), axis=1) if len(paused_ee) > 1 else np.array([np.nan]))
    em = _ee_step_metrics(d)
    return {
        "tag": run["tag"],
        "duration_s": float(t[-1] - t[0]),
        "n_ticks": s.get("n_ticks"),
        "n_popped": s.get("n_popped"),
        "n_ik_ok": s.get("n_ik_ok"),
        "n_ik_skip": s.get("n_ik_skip"),
        "n_underrun": s.get("n_underrun"),
        "n_paused": s.get("n_paused"),
        "frac_inferring_pct": round(100.0 * s.get("n_ik_ok", 0) / max(s.get("n_ticks", 1), 1), 1),
        "tick_dt_ms": s.get("tick_dt_ms"),
        "steady_dt_mean_ms": mean(steady_dt),
        "steady_dt_p99_ms": pct(steady_dt, 99),
        "steady_hz": 1000.0 / mean(steady_dt) if np.isfinite(mean(steady_dt)) else float("nan"),
        "spike_ms": float(d["tick_dt_ms"][idx]),
        "spike_at_s": float(t[idx]),
        "spike_at_tick": int(d["tick"][idx]),
        "ee_delta_m": s.get("ee_delta_m"),
        "joint_delta_max_deg": s.get("joint_delta_max_rad"),  # degrees (key mislabeled "_rad")
        "joint_tracking_deg_mean": float(mean(j_track)),
        "joint_tracking_deg_p95": float(pct(j_track, 95)),
        "joint_command_step_deg_mean": float(mean(j_cmd)),
        "joint_measured_step_deg_mean": float(mean(j_meas)),
        "n_valid": int(valid.sum()),
        "n_valid_segments": n_seg,
        "valid_duration_s": float(t[valid][-1] - t[valid][0]) if valid.any() else 0.0,
        "per_joint_step_deg": [round(float(v), 4) for v in per_joint],
        "exec_ee_step_mm_max": float(np.nanmax(exec_ee_step) * 1e3),
        "popped_paused_ee_step_mm_max": (float(np.nanmax(paused_step) * 1e3)
                                        if np.isfinite(paused_step).any() else float("nan")),
        "ee_step_mm_mean": float(np.mean(em["mag"])) if em["mag"].size else float("nan"),
        "ee_step_switch_mm_mean": (float(np.mean(em["mag_switch"])) if em["mag_switch"].size
                                   else float("nan")),
        "direction_reversal_rate": em["reversal"],
        "e2e_ms": s.get("e2e_ms"),
        "has_network_latency": bool(np.isfinite(d["e2e_ms"]).any()),
        "args_of_interest": {
            "pretrained_path": run["args"].get("pretrained_path"),
            "fps": run["args"].get("fps"),
            "n_action_steps": run["args"].get("n_action_steps"),
            "actions_per_chunk": run["args"].get("actions_per_chunk"),
            "chunk_size_threshold": run["args"].get("chunk_size_threshold"),
            "aggregate_fn_name": run["args"].get("aggregate_fn_name"),
            "warm_start": run["args"].get("warm_start"),
            "max_ee_step_m": run["args"].get("max_ee_step_m"),
        },
    }


def main() -> None:
    sync_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SYNC
    async_arg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ASYNC
    outdir = Path(sys.argv[3]) if len(sys.argv) > 3 else OUTDIR_DEFAULT
    if not outdir.is_absolute():
        outdir = REPO / outdir
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    S = load(resolve_stem(sync_arg))
    A = load(resolve_stem(async_arg))
    for run in (S, A):
        run["title"] = {
            "sync": "SYNC  deploy_umi_relative_ee_piper.py",
            "async": "ASYNC  async_umi_relative_ee_piper_client.py",
        }.get(run["tag"], run["tag"].upper())
    runs = [S, A]

    fig_ee_pose(runs, figdir)
    fig_joints(runs, figdir)
    fig_joint_decomposition(runs, figdir)
    fig_timing(runs, figdir)
    fig_stutter(runs, figdir)
    fig_execution_strategy(runs, figdir)

    sb, ab = block(S), block(A)
    out = {"sync": sb, "async": ab,
           "ratio_sync_async_joint_tracking_mean": sb["joint_tracking_deg_mean"] /
           ab["joint_tracking_deg_mean"],
           "sources": {"sync": str(resolve_stem(sync_arg).name), "async": str(resolve_stem(async_arg).name)}}
    (outdir / "computed_stats.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"Figures + stats written to {outdir}")
    print(json.dumps({
        "sync_valid": {"ticks": sb["n_valid"], "segs": sb["n_valid_segments"],
                       "j3_track_deg": sb["per_joint_step_deg"][2],
                       "exec_ee_max_mm": sb["exec_ee_step_mm_max"],
                       "phantom_paused_max_mm": sb["popped_paused_ee_step_mm_max"]},
        "async_valid": {"ticks": ab["n_valid"], "segs": ab["n_valid_segments"],
                        "j3_track_deg": ab["per_joint_step_deg"][2]},
        "joint_signals_deg_mean [track,cmd,meas]": {
            "sync": [round(sb["joint_tracking_deg_mean"], 3), round(sb["joint_command_step_deg_mean"], 3),
                     round(sb["joint_measured_step_deg_mean"], 3)],
            "async": [round(ab["joint_tracking_deg_mean"], 3), round(ab["joint_command_step_deg_mean"], 3),
                      round(ab["joint_measured_step_deg_mean"], 3)]},
        "ratio_tracking_sync_over_async": out["ratio_sync_async_joint_tracking_mean"],
    }, indent=2))


if __name__ == "__main__":
    main()

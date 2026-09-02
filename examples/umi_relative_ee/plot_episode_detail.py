#!/usr/bin/env python

"""Per-tick execution detail for one async UMI client run (`--log`).

Overlays, on the same time axis, everything that happened tick by tick:

  - achieved EE pose (FK of the simulated arm)            — thick dark line
  - commanded EE target per tick (raw popped target, PRE-clamp) — dots, chunk-colored
  - the bounds-CLAMPED target actually fed to IK (`action_ee_clamped`;
    grey dashed) — where it peels away from the dots, the workspace clamp
    is eating the command (EEBoundsAndSafety also warns on the console)
  - every incoming chunk's FULL predicted trajectory (30 absolute EE targets
    from `<stem>_chunks.npz`, including the tail later replaced) — thin lines,
    chunk-colored, so discarded plan tails are visible
  - tracking error (commanded vs achieved)
  - a chunk Gantt row: executed span (solid) vs intended 30-tick horizon
    (light) — how much of each prediction actually ran before a newer chunk
    replaced it; RTC replace chunks are outlined
  - tick-in-chunk index: how deep into its prediction each executed step sits

Chunk color = arrival order on a sequential ramp (time-ordered identity, not
a categorical series); achieved/commanded wear ink so the predictions stay
readable behind them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

INK = "0.12"
INK_SOFT = "0.35"
GRID = {"color": "0.9", "linewidth": 0.5}
AXIS_NAMES = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", required=True, help="control-log stem (needs .npz and _chunks.npz)")
    p.add_argument("--label", default="run", help="arm label for titles/filenames (e.g. rtc)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out_dir", default=None, help="default: alongside the log")
    p.add_argument("--zoom", default="auto", help="'auto' (busiest 4 s) or 'START:END' seconds since engage")
    return p.parse_args()


def _derive_clamped(stem: Path, action_ee: np.ndarray) -> np.ndarray | None:
    """Backfill the bounds-clipped target for logs predating `action_ee_clamped`.

    Exact for executed ticks: EEBoundsAndSafety clips first, then REJECTS any
    post-clip jump — so an executed tick's clamped target is precisely
    clip(raw, ee_bounds), with the bounds from the run's own recorded args.
    """
    meta_path = Path(f"{stem}.json")
    if not meta_path.exists():
        return None
    run_args = json.loads(meta_path.read_text()).get("args", {})
    lo, hi = run_args.get("ee_bounds_min"), run_args.get("ee_bounds_max")
    if lo is None or hi is None:
        return None
    out = action_ee.copy()
    out[:, :3] = np.clip(out[:, :3], lo, hi)
    return out


def load(stem: Path) -> dict:
    d = np.load(f"{stem}.npz", allow_pickle=True)
    chunks = np.load(f"{stem}_chunks.npz")
    step = d["step"].astype(float)
    at = d["action_timestep"].astype(float)
    cid = d["chunk_id"].astype(float)
    executed = d["popped"].astype(bool) & d["ik_ok"].astype(bool)
    engaged_idx = int(np.argmax(executed)) if executed.any() else len(step)
    action_ee = d["action_ee"].astype(float)
    cmd_clamped = (
        d["action_ee_clamped"].astype(float) if "action_ee_clamped" in d.files
        else _derive_clamped(stem, action_ee)
    )

    # timestep -> step linear map from executed ticks (one action per timestep,
    # one timestep per tick, so slope ≈ 1; regression absorbs underrun gaps).
    ts, ss = at[executed], step[executed]
    slope, intercept = np.polyfit(ts, ss, 1) if len(ts) >= 2 else (1.0, float(ss[0]) - float(ts[0]))

    first_ts = chunks["first_ts"].astype(float)
    n = chunks["n"].astype(int)
    replace = chunks["replace"].astype(bool)
    pred = chunks["abs"].astype(float)  # [N, max_n, 7]
    # Executed span per chunk row: rows arrive in anchor order and executed
    # chunk_ids in the same order (the client counter increments per accepted
    # merge), so pair them by order and validate by timestep containment — a
    # chunk's first EXECUTED timestep is typically first_ts + inference_delay
    # (the in-transit prefix was already executed and gets dropped on arrival).
    exec_span = np.full((len(first_ts), 2), np.nan)
    cids = np.unique(cid[executed & np.isfinite(cid)])
    for row, c in zip(range(len(first_ts)), cids, strict=False):
        m = executed & (cid == c)
        lo, hi = float(at[m].min()), float(at[m].max())
        if first_ts[row] - 0.5 <= lo and hi <= first_ts[row] + max(n[row] - 1, 0) + 0.5:
            exec_span[row] = [step[m].min(), step[m].max()]

    return {
        "step": step, "at": at, "cid": cid, "executed": executed,
        "cmd": action_ee, "cmd_clamped": cmd_clamped,
        "ach": d["current_ee"].astype(float),
        "queue": d["queue"].astype(float), "engaged_idx": engaged_idx,
        "slope": slope, "intercept": intercept,
        "first_ts": first_ts, "n": n, "replace": replace, "pred": pred,
        "exec_span": exec_span,
    }


def t2s(run: dict, t: np.ndarray | float) -> np.ndarray:
    return run["intercept"] + run["slope"] * np.asarray(t, dtype=float)


def time_s(run: dict, fps: int) -> np.ndarray:
    return (run["step"] - run["step"][run["engaged_idx"]]) / fps


def s2t(run: dict, step_values, fps: int) -> np.ndarray:
    """Control-step values -> seconds since engage (the figure's time base)."""
    return (np.asarray(step_values, dtype=float) - run["step"][run["engaged_idx"]]) / fps


def draw_detail(run: dict, label: str, fps: int, s_lo: float, s_hi: float,
                title_suffix: str, out: Path) -> dict:
    """Render the 6-row detail figure for the step window [s_lo, s_hi]."""
    step, ach, cmd = run["step"], run["ach"], run["cmd"]
    cc = run.get("cmd_clamped")
    executed = run["executed"]
    origin = ach[run["engaged_idx"], :3]
    t = time_s(run, fps)
    win = (step >= s_lo) & (step <= s_hi)
    t_lo, t_hi = s2t(run, s_lo, fps), s2t(run, s_hi, fps)
    # chunks overlapping the window (by intended horizon, so tails entering the
    # window from an earlier anchor are included)
    anchor_s = t2s(run, run["first_ts"])
    end_s = t2s(run, run["first_ts"] + np.maximum(run["n"] - 1, 0))
    chunk_sel = np.where((end_s >= s_lo) & (anchor_s <= s_hi))[0]
    cmap = plt.get_cmap("viridis_r")
    norm = Normalize(vmin=0, vmax=max(1, len(run["first_ts"]) - 1))

    fig, axes = plt.subplots(
        6, 1, figsize=(12.0, 11.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 3, 2.2, 2.6, 2.0]},
    )

    for k, ax in enumerate(axes[:3]):
        # full predicted chunk trajectories (thin, chunk-colored)
        for i in chunk_sel:
            t0, n_i = run["first_ts"][i], run["n"][i]
            s_pred = t2s(run, t0 + np.arange(n_i))
            y_pred = (run["pred"][i, :n_i, k] - origin[k]) * 1000.0
            m = (s_pred >= s_lo - 1) & (s_pred <= s_hi + 1) & np.isfinite(y_pred)
            if m.any():
                ax.plot(s2t(run, s_pred[m], fps), y_pred[m], color=cmap(norm(i)),
                        alpha=0.45, linewidth=1.0, zorder=2)
        ax.plot(t[win], (ach[win, k] - origin[k]) * 1000.0, color=INK, linewidth=1.9,
                zorder=4, label="achieved (FK)" if k == 0 else None)
        if cc is not None:
            mc = win & executed & np.isfinite(cc[:, k])
            ax.plot(t[mc], (cc[mc, k] - origin[k]) * 1000.0, color=INK_SOFT,
                    linewidth=1.1, linestyle=(0, (4, 2)), zorder=3,
                    label="clamped cmd (to IK)" if k == 0 else None)
        m = win & executed & np.isfinite(cmd[:, k])
        ax.scatter(t[m], (cmd[m, k] - origin[k]) * 1000.0, s=13, zorder=5,
                   c=[cmap(norm(i)) for i in
                      (np.searchsorted(run["first_ts"], run["at"][m]) - 1).clip(0)],
                   edgecolor="white", linewidth=0.4,
                   label="commanded / tick" if k == 0 else None)
        ax.set_ylabel(f"EE {AXIS_NAMES[k]} [mm]")
        if k == 0:
            ax.legend(fontsize=8, frameon=False, loc="best", ncol=2)

    # tracking error: vs the raw commanded target (policy intent) and vs the
    # clamped target (true control error — the gap between the two curves is
    # the part the workspace clamp ate)
    err_mm = np.linalg.norm(cmd[:, :3] - ach[:, :3], axis=1) * 1000.0
    m = win & executed
    axes[3].plot(t[m], err_mm[m], color=INK, linewidth=1.1,
                 label="vs raw cmd (policy intent)")
    if cc is not None:
        err_clamped = np.linalg.norm(cc[:, :3] - ach[:, :3], axis=1) * 1000.0
        m2 = m & np.isfinite(err_clamped)
        axes[3].plot(t[m2], err_clamped[m2], color=INK_SOFT, linewidth=1.0,
                     linestyle=(0, (4, 2)), label="vs clamped cmd (control)")
        axes[3].legend(fontsize=8, frameon=False, loc="upper left")
    axes[3].set_ylabel("tracking err\n[mm]")

    # chunk Gantt: light bar = intended 30-tick horizon, solid = executed part
    ax = axes[4]
    for row, i in enumerate(chunk_sel[::-1]):
        y = row
        a0, a1 = s2t(run, anchor_s[i], fps), s2t(run, end_s[i], fps)
        ax.barh(y, a1 - a0, left=a0, height=0.62, color=cmap(norm(i)), alpha=0.22,
                edgecolor=cmap(norm(i)), linewidth=0.6,
                linestyle="--" if run["replace"][i] else "-",
                zorder=2)
        e0, e1 = run["exec_span"][i]
        if np.isfinite(e0):
            e0, e1 = s2t(run, max(e0, s_lo), fps), s2t(run, min(e1, s_hi), fps)
            if e0 > a0:  # in-transit prefix: skipped, already executed on arrival
                ax.barh(y, e0 - a0, left=a0, height=0.62, color="0.72", zorder=2.5)
            ax.barh(y, e1 - e0, left=e0, height=0.62,
                    color=cmap(norm(i)), zorder=3)
    ax.set_yticks(range(len(chunk_sel)),
                  [f"chunk {i}{' (RTC)' if run['replace'][i] else ''}" for i in chunk_sel[::-1]],
                  fontsize=7)
    ax.set_ylabel("incoming chunks")
    # executed-step colorbar trick: leave ticks to the tick-in-chunk row below

    # tick-in-chunk index: how deep into its prediction each executed step sits
    ax = axes[5]
    m = win & executed
    owner = (np.searchsorted(run["first_ts"], run["at"][m]) - 1).clip(0)
    depth = np.full(m.sum(), np.nan)
    for j, i in enumerate(owner):
        if 0 <= i < len(run["first_ts"]):
            depth[j] = run["at"][m][j] - run["first_ts"][i]
    ax.scatter(t[m], depth, s=12, c=[cmap(norm(i)) for i in owner],
               edgecolor="white", linewidth=0.3, zorder=3)
    ax.axhline(run["n"].max() - 1, color=INK_SOFT, linewidth=0.8, linestyle="--")
    ax.annotate(f"prediction horizon ({run['n'].max()} ticks)", (0.995, run["n"].max() - 1),
                xycoords=("axes fraction", "data"), xytext=(0, 3),
                textcoords="offset points", ha="right", fontsize=8, color=INK_SOFT)
    ax.set_ylabel("tick index\nwithin chunk")
    ax.set_xlabel("time since engage [s]")

    axes[0].set_xlim(t_lo, t_hi)
    for ax in axes:
        ax.grid(True, **GRID)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("", loc="center")
    # wrap=True breaks the legend at the axes width, so a long parenthetical
    # can never clip past the figure's right edge.
    axes[0].set_title(
        f"Per-tick execution detail — {label}{title_suffix}  "
        "(thin colored = full chunk prediction incl. replaced tail; thick dark = achieved; "
        "dots = commanded/tick; grey dashed = bounds-clamped cmd to IK; Gantt: hollow = "
        "intended 30-tick horizon, grey = in transit/skipped, solid = executed, "
        "dashed = RTC replace)",
        fontsize=9.5, color=INK, loc="left", wrap=True,
    )
    # chunk-order colorbar
    sm = ScalarMappable(norm=Normalize(0, max(1, len(run["first_ts"]) - 1)), cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.028, pad=0.008)
    cb.set_label("chunk arrival order", fontsize=8)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    used = (run["exec_span"][chunk_sel, 1] - run["exec_span"][chunk_sel, 0]) + 1
    intended = run["n"][chunk_sel]
    info = {
        "window_steps": [float(s_lo), float(s_hi)],
        "n_chunks": int(len(chunk_sel)),
        "n_replace": int(run["replace"][chunk_sel].sum()),
        "executed_fraction_mean": float(np.nanmean(used / intended)) if len(used) else float("nan"),
    }
    if cc is not None:
        mexec = (step >= s_lo) & (step <= s_hi) & executed
        excess = np.linalg.norm(cmd[mexec, :3] - cc[mexec, :3], axis=1) * 1000.0
        excess = excess[np.isfinite(excess)]
        if excess.size:
            info["clamp_active_pct"] = float((excess > 1.0).mean() * 100.0)
            info["clamp_excess_max_mm"] = float(excess.max())
    return info


def main() -> None:
    args = parse_args()
    stem = Path(args.log)
    run = load(stem)
    out_dir = Path(args.out_dir) if args.out_dir else stem.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    s_lo, s_hi = run["step"][run["engaged_idx"]], run["step"][-1]
    full = draw_detail(run, args.label, args.fps, s_lo, s_hi, " (full run)",
                       out_dir / f"episode_detail_{args.label}.png")

    # Zoom window: DETERMINISTIC (engage + 0.5 s, 4 s wide, clamped to the run)
    # so the two arms' zoom figures share the exact same time span and can be
    # compared side by side — a per-arm "busiest window" pick would desync them.
    if args.zoom.lower() == "auto":
        z_lo = s_lo + round(0.5 * args.fps)
        z_hi = min(z_lo + round(4.0 * args.fps), s_hi - 3)
    else:
        a, b = args.zoom.split(":", 1)
        z_lo = s_lo + float(a) * args.fps
        z_hi = s_lo + float(b) * args.fps
    zoom = draw_detail(
        run, args.label, args.fps, z_lo, z_hi,
        f" (zoom {(z_lo - s_lo) / args.fps:.1f}–{(z_hi - s_lo) / args.fps:.1f} s)",
        out_dir / f"episode_detail_{args.label}_zoom.png",
    )

    summary = {"label": args.label, "log": str(stem), "full": full, "zoom": zoom}
    (out_dir / f"episode_detail_{args.label}.json").write_text(json.dumps(summary, indent=2))
    print(f"{args.label}: {full['n_chunks']} chunks in window "
          f"({full['n_replace']} RTC-replace), executed fraction of predictions: "
          f"{full['executed_fraction_mean'] * 100:.0f}%")
    print(f"Wrote {out_dir}/episode_detail_{args.label}{{,_zoom}}.png")


if __name__ == "__main__":
    main()

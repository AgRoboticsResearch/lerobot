#!/usr/bin/env python
"""Offline experiment: does UMI-style Cartesian trajectory interpolation help our
relative-EE ACT deploy?

WHY THIS EXPERIMENT
-------------------
UMI (real-stanford/universal_manipulation_interface) does NOT step a policy's actions
verbatim into the arm. It treats the policy output as **timestamped Cartesian EE
waypoints**, builds a continuous trajectory (linear XYZ + SLERP rotation), samples it
in a fast controller loop, and -- crucially -- on every re-plan **splices the new chunk
onto the currently *commanded* trajectory, not the measured robot pose**. UMI's own
comment warns that anchoring the next chunk at the lagged measured pose causes
"discontinuity and jittery robot behavior."

Our SYNC deploy *does* anchor each new relative chunk at the measured pose and replay
it verbatim. We proved (real-deploy --log) that the SYNC chunk-switch jump equals the
tracking gap (arm lags ~33 ms commands). So the question is whether UMI's interpolation
+ splice removes that jump and otherwise helps.

WHAT WE SIMULATE (offline, ACT 1459 on validation episodes)
-----------------------------------------------------------
The policy is run once per episode on the recorded GT observation stream. Its predicted
chunks are then frozen and shared byte-for-byte by every execution strategy. A separate
executor simulation applies a configurable pure command-tracking delay of L ticks. This
isolates execution and anchoring from policy-feedback effects.

  1. sync_replay        -- our baseline: anchor at measured, pop verbatim at 30 Hz.
  2. umi_splice         -- faithful UMI execution analog: keep measured-decoded absolute
                           targets, splice to them from the commanded trajectory.
  3. cmd_reanchor       -- experimental continuity hack: rigidly re-anchor the whole
                           relative chunk at the commanded trajectory value.
  4. async_ensemble     -- reference: overlapping chunks, weighted_average blend.

For this checkpoint action index 0 is the current pose target (the training window is
[-1, 0, ..., 29] and the processor removes -1). The UMI executors therefore discard
index 0 as stale/current and schedule index i at now+i*dt, matching UMI's timestamping.

Because our action spacing (1/30 s) == our control rate (30 Hz), there is no native
densification (unlike UMI's 10 Hz->200 Hz); we also test interpolation-as-lowpass by
optionally subsampling the chunk to sparse waypoints before interpolating.

OUTPUTS
-------
  01_two_chunk_detail.png   -- chunk1->chunk2 transition, GT + waypoints + each strategy
                               + measured, with a switch zoom-inset (THE figure).
  02_switch_jumps.png        -- boundary velocity-change magnitude (legacy filename).
  03_trajectory_overview.png -- full-episode commanded vs GT (x,y,z) per strategy.
  04_jitter_drift.png        -- per-tick step + reversal + drift-vs-GT.
  metrics.json + this report's tables.

Usage:
  uv run python examples/umi_relative_ee/control_debug/umi_interp_experiment.py \
      --pretrained_path outputs/train/act_umi_identity_rot6d_1459/checkpoints/last/pretrained_model \
      --dataset /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
      --episodes 0 1 2 3 4 --lag_ticks 3 \
      --out outputs/research_report/low_level_control_debug/umi_interp_experiment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# examples/umi_relative_ee/ is the parent of this file's dir -> import siblings.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Reuse the sync mock's policy loader + helpers (real deploy path).
from mock_sync_client import _ee7, add_policy_task, load_policy_and_processors
from umi_trajectory_interpolator import PoseTrajectoryInterpolator7D

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Relative<->absolute re-anchoring (so we can anchor a chunk at the commanded pose).
from lerobot.processor.umi_relative_ee_processor import (
    absolute_aa_to_relative_rot6d,
    relative_rot6d_to_absolute_aa,
)
from lerobot.utils.constants import OBS_STATE

CAMERA_KEY = "observation.images.camera"
DIM_TITLES = ["x [m]", "y [m]", "z [m]", "rot_x [rad]", "rot_y [rad]", "rot_z [rad]", "gripper"]
STRATEGY_COLORS = {
    "sync_replay": "tab:red",
    "umi_splice": "tab:purple",
    "cmd_reanchor": "tab:green",
    "async_ensemble": "tab:brown",
}
STRATEGY_LABELS = {
    "sync_replay": "SYNC replay (anchor=measured, verbatim)",
    "umi_splice": "UMI splice (absolute targets decoded at measured)",
    "cmd_reanchor": "Command-reanchored relative chunk (experimental)",
    "async_ensemble": "ASYNC ensemble (weighted_average)",
}


# --------------------------------------------------------------------------------------
# Strategy executors -- each turns a predicted absolute chunk into a commanded trajectory.
# --------------------------------------------------------------------------------------
class SyncReplay:
    """Open-loop chunk replay (our SYNC baseline): pop verbatim at the control rate."""

    name = "sync_replay"

    def __init__(self, replan_every=30):
        self.replan_every = replan_every
        self.chunk: list[np.ndarray] = []
        self.idx = 0

    def replan(self, t, now, abs_meas, abs_cmd, measured, commanded_now, **_):
        self.chunk = [np.asarray(a, dtype=np.float64) for a in abs_meas]
        self.idx = 0

    def execute(self, t, now):
        if self.idx >= len(self.chunk):
            return None
        a = self.chunk[self.idx]
        self.idx += 1
        return a


class UmiInterp:
    """Schedule waypoints into the interpolator; sample at the control rate.

    ``anchor`` selects which absolute chunk is scheduled: ``measured`` preserves the
    policy's measured-decoded absolute targets (the faithful UMI execution analog), while
    ``commanded`` rigidly re-anchors the complete relative plan (an experimental variant).
    ``waypoint_stride`` subsamples the chunk to sparse waypoints (interpolation-as-lowpass).
    """

    def __init__(
        self,
        anchor: str,
        replan_every=30,
        waypoint_stride: int = 1,
        max_pos_speed=np.inf,
        max_rot_speed=np.inf,
    ):
        self.replan_every = replan_every
        self.anchor = anchor
        self.stride = max(1, waypoint_stride)
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.interp: PoseTrajectoryInterpolator7D | None = None
        self.last_wp_time = None

    def replan(self, t, now, abs_meas, abs_cmd, measured, commanded_now, dt, **_):
        chunk = abs_cmd if self.anchor == "commanded" else abs_meas
        chunk = np.asarray(chunk, dtype=np.float64)
        # Index 0 is the current-pose target for this checkpoint and is stale by the time
        # inference completes. Preserve the active command at ``now`` and schedule future
        # targets at their original action timestamps.
        if len(chunk) < 2:
            return
        idxs = list(range(1, len(chunk), self.stride))
        if idxs[-1] != len(chunk) - 1:
            idxs.append(len(chunk) - 1)
        wps = chunk[idxs]
        if self.interp is None:
            self.interp = PoseTrajectoryInterpolator7D([now], [commanded_now])
        for idx, wp in zip(idxs, wps, strict=True):
            ti = float(now + idx * dt)
            self.interp = self.interp.schedule_waypoint(
                wp,
                ti,
                curr_time=now,
                last_waypoint_time=self.last_wp_time,
                max_pos_speed=self.max_pos_speed,
                max_rot_speed=self.max_rot_speed,
            )
            self.last_wp_time = ti

    def execute(self, t, now):
        return self.interp(now)


class AsyncEnsemble:
    """Reference: overlapping chunks blended by a recency-weighted average of absolutes."""

    name = "async_ensemble"

    def __init__(self, replan_every=10, chunk_len=20, blend=0.3):
        self.replan_every = replan_every
        self.chunk_len = chunk_len
        self.blend = blend  # weight on the OLDER (existing) target; 0.3 like the deploy
        self.active: list[tuple[int, np.ndarray]] = []  # (replan_tick, abs_chunk)

    def replan(self, t, now, abs_meas, abs_cmd, measured, commanded_now, **_):
        if t % self.replan_every == 0:
            self.active.append((t, np.asarray(abs_meas, dtype=np.float64)))
            # drop chunks fully in the past
            self.active = [(tr, c) for (tr, c) in self.active if t - tr < self.chunk_len]

    def execute(self, t, now):
        contribs = []
        for tr, chunk in self.active:
            k = t - tr
            if 0 <= k < len(chunk):
                contribs.append((t - tr, chunk[k]))
        if not contribs:
            return None
        contribs.sort(key=lambda x: x[0])  # oldest -> newest
        # incremental blend: newer chunks weigh (1-blend); emulate ActionBuffer order
        agg = contribs[0][1].copy()
        for _age, a in contribs[1:]:
            agg = self.blend * agg + (1.0 - self.blend) * a
        return agg


# --------------------------------------------------------------------------------------
# Closed-loop lag simulator
# --------------------------------------------------------------------------------------
def _reanchor(abs_meas_np, measured_np, commanded_np):
    """Vectorised re-anchor of an (N,7) absolute chunk from measured -> commanded pose."""
    meas = torch.as_tensor(measured_np, dtype=torch.float32)
    cmd = torch.as_tensor(commanded_np, dtype=torch.float32)
    abso = torch.as_tensor(abs_meas_np, dtype=torch.float32)  # (N,7)
    meas_e = meas.unsqueeze(0).expand(abso.shape[0], 7)  # (N,7)
    cmd_e = cmd.unsqueeze(0).expand(abso.shape[0], 7)  # (N,7)
    rel = absolute_aa_to_relative_rot6d(meas_e, abso)  # (N,10)
    return relative_rot6d_to_absolute_aa(rel, cmd_e).numpy()  # (N,7)


def collect_shared_predictions(args, episode, policy, preprocessor, postprocessor, device, prediction_every):
    """Run the policy once on GT observations and cache immutable absolute chunks.

    The preprocessor is called on *every* tick, as in live deployment, so its two-frame
    relative-state cache contains t-1 and t rather than the previous replan observation.
    """
    ds = LeRobotDataset(args.dataset, episodes=[episode], root=args.root)
    n = len(ds)
    gt = np.zeros((n, 7), dtype=np.float64)
    chunks: dict[int, np.ndarray] = {}

    preprocessor.reset()
    postprocessor.reset()
    policy.reset()
    for t in range(n):
        sample = ds[t]
        pose = _ee7(sample[args.ee_key])
        gt[t] = pose
        with torch.no_grad():
            state = torch.from_numpy(pose).unsqueeze(0).float().to(device)
            batch = {OBS_STATE: state, CAMERA_KEY: sample[CAMERA_KEY].unsqueeze(0).to(device)}
            add_policy_task(batch, policy, args.task)
            processed = preprocessor(batch)
            if t % prediction_every != 0:
                continue
            if policy.config.type == "act":
                processed.pop("action", None)
            pred_norm = policy.predict_action_chunk(processed)
            pred = postprocessor(pred_norm)
            if isinstance(pred, dict) and "action" in pred:
                pred = pred["action"]
            chunks[t] = pred[0, : args.n_action_steps].cpu().numpy().astype(np.float64)
    return {"episode": episode, "gt": gt, "chunks": chunks, "n": n, "prediction_every": prediction_every}


def run_strategy(args, shared, strategy, lag_ticks):
    """Execute shared policy chunks under one anchoring/scheduling strategy."""
    gt = shared["gt"]
    n = shared["n"]
    replan_every = strategy.replan_every
    dt = 1.0 / args.fps

    commanded = [None] * n
    measured = [None] * n
    chunk_starts: list[int] = []  # tick where each replan happened
    chunks_meas: list[np.ndarray] = []  # the predicted absolute chunk (anchor=measured)
    chunks_cmd: list[np.ndarray] = []  # experimental command-reanchored chunks
    replan_anchor_gap_mm: list[float] = []
    seed0 = gt[0].copy()

    for t in range(n):
        now = t * dt
        # ---- tracking-lag model: measured = commanded delayed by L ticks ----
        if t == 0:
            m = seed0.copy()
        else:
            src = t - lag_ticks
            m = commanded[src] if src >= 0 and commanded[src] is not None else seed0.copy()
        measured[t] = m

        # ---- re-plan on cadence ----
        if t % replan_every == 0:
            chunk_starts.append(t)
            abs_gt = shared["chunks"][t]
            # Decode the exact same relative prediction at the simulated measured pose.
            abs_meas = _reanchor(abs_gt, gt[t], m)
            # commanded_now for re-anchoring = current commanded trajectory value
            commanded_now = (
                strategy.execute(t, now)
                if _has_execute_now(strategy)
                else (commanded[t - 1] if t > 0 else m.copy())
            )
            if commanded_now is None:
                commanded_now = m.copy()
            replan_anchor_gap_mm.append(ee_xyz_mm(commanded_now, m))
            abs_cmd = _reanchor(abs_meas, m, commanded_now)
            chunks_meas.append(abs_meas)
            chunks_cmd.append(abs_cmd)
            strategy.replan(
                t=t,
                now=now,
                abs_meas=abs_meas,
                abs_cmd=abs_cmd,
                measured=m,
                commanded_now=commanded_now,
                dt=dt,
            )

        # ---- execute ----
        c = strategy.execute(t, now)
        if c is None:  # ran out (e.g. sync queue drained early) -> hold last
            c = commanded[t - 1] if t > 0 else seed0.copy()
        commanded[t] = np.asarray(c, dtype=np.float64)

    commanded = np.stack(commanded)
    measured = np.stack(measured)
    return {
        "commanded": commanded,
        "measured": measured,
        "gt": gt,
        "dt": dt,
        "chunk_starts": chunk_starts,
        "chunks_meas": chunks_meas,
        "chunks_cmd": chunks_cmd,
        "lag_ticks": lag_ticks,
        "replan_every": replan_every,
        "n": n,
        "episode": shared["episode"],
        "replan_anchor_gap_mm": replan_anchor_gap_mm,
    }


def _has_execute_now(strategy):
    # only UMI can be sampled at an arbitrary 'now' before its first replan; for sync/async
    # execute depends on having called replan first. We always replan at t%R==0 (incl t=0)
    # before execute, so commanded_now from a prior execute is fine for t>0.
    return isinstance(strategy, UmiInterp) and strategy.interp is not None


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def ee_xyz_mm(a, b):
    return np.linalg.norm(a[:3] - b[:3]) * 1000.0


def reversal_rate(cmd_xyz):
    """Fraction of executed ticks whose step reverses direction vs the previous step."""
    d = np.diff(cmd_xyz, axis=0)
    if len(d) < 2:
        return 0.0, np.zeros(len(cmd_xyz))
    v = d[:-1]
    w = d[1:]
    norms = np.linalg.norm(v, axis=1) * np.linalg.norm(w, axis=1)
    dot = (v * w).sum(axis=1)
    cos = np.where(norms > 1e-12, dot / np.clip(norms, 1e-12, None), 1.0)
    rev = cos < 0.0
    # mark reversal at the second tick of each pair (length len(cmd)-2); pad to len(cmd)
    flags = np.zeros(len(cmd_xyz), dtype=bool)
    flags[2:] = rev
    return float(rev.mean()), flags


def mean_second_difference(cmd):
    """Mean translational |second difference| in metres/tick²."""
    d2 = np.diff(cmd[:, :3], n=2, axis=0)
    return float(np.mean(np.linalg.norm(d2, axis=1)))


def switch_boundaries(commanded, chunk_starts, window=2):
    """Return boundary step and velocity-change magnitudes in millimetres.

    ``velocity_change`` is the maximum change between adjacent command steps from the
    boundary through ``window`` following ticks. Unlike the old maximum-step metric, it
    measures a velocity discontinuity rather than labeling ordinary speed as a jump.
    """
    out = []
    cmd = np.asarray(commanded)
    steps = np.zeros(len(cmd))
    steps[1:] = np.linalg.norm(np.diff(cmd[:, :3], axis=0), axis=1) * 1000.0
    for s in chunk_starts:
        if s <= 0:
            continue
        step = float(steps[s])
        if s >= 2:
            diffs = np.diff(cmd[:, :3], axis=0)
            lo = s - 1
            hi = min(len(diffs) - 1, s - 1 + window)
            accelerations = diffs[lo : hi + 1] - diffs[lo - 1 : hi]
            velocity_change = float(np.linalg.norm(accelerations, axis=1).max() * 1000.0)
        else:
            velocity_change = 0.0
        out.append((s, step, velocity_change))
    return out


def _per_tick_steps_mm(cmd):
    return np.concatenate([[0.0], np.linalg.norm(np.diff(cmd[:, :3], axis=0), axis=1)]) * 1000.0


def summarize(run):
    cmd = run["commanded"]
    gt = run["gt"]
    n = min(len(cmd), len(gt))
    drift = [ee_xyz_mm(cmd[t], gt[t]) for t in range(n)]
    rev_rate, _ = reversal_rate(cmd[:, :3])
    boundaries = switch_boundaries(cmd, run["chunk_starts"])
    boundary_steps = [step for _, step, _ in boundaries]
    boundary_velocity_changes = [dv for _, _, dv in boundaries]
    mean_step = float(np.mean(np.linalg.norm(np.diff(cmd[:n, :3], axis=0), axis=1)) * 1000)
    tracking_gap = np.linalg.norm(cmd[:n, :3] - run["measured"][:n, :3], axis=1) * 1000.0
    return {
        "n_ticks": n,
        "drift_mean_mm": float(np.mean(drift)),
        "drift_max_mm": float(np.max(drift)),
        "reversal_rate": rev_rate,
        "mean_step_mm": mean_step,
        "second_difference_m_per_tick2": mean_second_difference(cmd),
        "tracking_gap_mean_mm": float(np.mean(tracking_gap)),
        "tracking_gap_max_mm": float(np.max(tracking_gap)),
        "replan_anchor_gap_mean_mm": float(np.mean(run["replan_anchor_gap_mm"])),
        "replan_anchor_gap_max_mm": float(np.max(run["replan_anchor_gap_mm"])),
        "boundary_step_mm": [round(x, 3) for x in boundary_steps],
        "boundary_velocity_change_mm": [round(x, 3) for x in boundary_velocity_changes],
        "boundary_step_mean_mm": float(np.mean(boundary_steps)) if boundary_steps else 0.0,
        "boundary_velocity_change_mean_mm": (
            float(np.mean(boundary_velocity_changes)) if boundary_velocity_changes else 0.0
        ),
        "n_switches": len(boundaries),
    }


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def _style_ax(ax, title):
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=7)


def plot_two_chunk_detail(runs, args, out):
    """THE figure: chunk1->chunk2 transition for one episode, all dims + switch zoom."""
    # find an episode/dim layout
    ep = args.detail_episode
    names = list(STRATEGY_LABELS)
    present = [nm for nm in names if (nm, ep) in runs]
    run0 = runs[(present[0], ep)]
    starts = run0["chunk_starts"]
    replan_every = run0["replan_every"]
    if len(starts) < 2:
        print(f"[detail] episode {ep} has <2 chunk starts; skipping detail figure")
        return
    s0, s1 = starts[1], starts[2] if len(starts) > 2 else run0["n"]
    win0 = max(0, s0 - replan_every)
    win1 = min(run0["n"], s1 + 3)
    xs = np.arange(win0, win1)
    dims = [0, 1, 2, 3, 6]  # x, y, z, rot_x, gripper
    fig, axes = plt.subplots(len(dims), 1, figsize=(11, 2.6 * len(dims)), sharex=True)
    for ax, d in zip(axes, dims, strict=True):
        gt = run0["gt"][win0:win1, d]
        ax.plot(xs, gt, "--", color="tab:orange", lw=1.8, label="GT", zorder=5)
        # policy waypoints (measured-anchored chunk) for the two chunks, as markers
        for ci, cs in enumerate([s for s in starts if win0 <= s < win1]):
            ck = run0["chunks_meas"][[i for i, s in enumerate(starts) if s == cs][0]][:replan_every, d]
            ax.plot(
                np.arange(cs, cs + len(ck)),
                ck,
                "o",
                color="0.6",
                ms=2.5,
                label="policy waypoints" if ci == 0 else None,
            )
        # measured (lagged) trace
        ax.plot(xs, run0["measured"][win0:win1, d], ":", color="0.35", lw=1.1, label="measured (lagged)")
        for nm in present:
            run = runs[(nm, ep)]
            ax.plot(
                xs,
                run["commanded"][win0:win1, d],
                "-",
                lw=1.6,
                color=STRATEGY_COLORS[nm],
                alpha=0.9,
                label=STRATEGY_LABELS[nm],
            )
        ax.axvline(s0, color="tab:red", ls="--", lw=1.0)
        if s1 < run0["n"]:
            ax.axvline(s1, color="tab:red", ls=":", lw=0.8)
        ax.set_ylabel(DIM_TITLES[d], fontsize=8)
        _style_ax(ax, "")
    axes[0].legend(fontsize=6.5, loc="upper right", ncol=2)
    axes[-1].set_xlabel("tick (30 Hz)")
    fig.suptitle(
        f"Two-chunk transition (episode {ep}, switch @tick {s0}, lag={run0['lag_ticks']} ticks)\n"
        f"red dashed = chunk re-plan (SYNC anchors new chunk at the lagged measured pose here)",
        fontsize=10,
    )
    # zoom inset on xyz at the switch
    ax_ins = fig.add_axes([0.62, 0.62, 0.35, 0.16])
    z0, z1 = s0 - 3, s0 + 4
    zx = np.arange(z0, z1)
    ax_ins.plot(zx, run0["gt"][z0:z1, 0], "--", color="tab:orange", lw=1.5)
    for nm in present:
        ax_ins.plot(zx, runs[(nm, ep)]["commanded"][z0:z1, 0], "-", color=STRATEGY_COLORS[nm], lw=1.5)
    ax_ins.axvline(s0, color="tab:red", ls="--", lw=0.8)
    ax_ins.set_title("zoom: EE-x around the switch", fontsize=7)
    ax_ins.tick_params(labelsize=6)
    ax_ins.grid(True, alpha=0.3, linewidth=0.5)
    fig.savefig(out / "01_two_chunk_detail.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig] 01_two_chunk_detail.png (episode {ep}, switch @tick {s0})")


def plot_switch_jumps(runs, out):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = [nm for nm in STRATEGY_LABELS if any((nm, e) in runs for e in range(99))]
    data = {nm: [] for nm in names}
    for (nm, _ep), run in runs.items():
        for _, _step, velocity_change in switch_boundaries(run["commanded"], run["chunk_starts"]):
            data[nm].append(velocity_change)
    positions, labels = [], []
    vals = []
    for i, nm in enumerate(names):
        js = data[nm]
        if not js:
            continue
        for j in js:
            positions.append(i)
            vals.append(j)
        ax.scatter(
            [i] * len(js), js, color=STRATEGY_COLORS[nm], alpha=0.55, s=28, zorder=3, edgecolors="none"
        )
        if js:
            ax.plot([i - 0.25, i + 0.25], [np.mean(js)] * 2, color=STRATEGY_COLORS[nm], lw=2.2, zorder=4)
        labels.append(f"{nm}\nmean={np.mean(js):.2f}mm\n(n={len(js)})" if js else nm)
    ax.scatter(positions, vals, color="0.85", s=1, zorder=1)  # keep axis
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("boundary velocity change  |Δcmd[s] − Δcmd[s−1]|  [mm/tick]")
    ax.set_title("Per-switch velocity discontinuity (all episodes). Lower = smoother stitching.")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "02_switch_jumps.png", dpi=150)
    plt.close(fig)
    print("[fig] 02_switch_jumps.png")


def plot_trajectory_overview(runs, args, out):
    ep = args.detail_episode
    names = [nm for nm in STRATEGY_LABELS if (nm, ep) in runs]
    run0 = runs[(names[0], ep)]
    n = run0["n"]
    xs = np.arange(n)
    dims = [0, 1, 2]
    fig, axes = plt.subplots(len(dims), 1, figsize=(11, 6), sharex=True)
    for ax, d in zip(axes, dims, strict=True):
        ax.plot(xs, run0["gt"][:, d], "--", color="tab:orange", lw=1.8, label="GT")
        for nm in names:
            ax.plot(
                xs,
                runs[(nm, ep)]["commanded"][:, d],
                "-",
                lw=1.3,
                color=STRATEGY_COLORS[nm],
                alpha=0.85,
                label=STRATEGY_LABELS[nm],
            )
        for s in run0["chunk_starts"][1:]:
            ax.axvline(s, color="tab:red", ls=":", lw=0.6, alpha=0.5)
        ax.set_ylabel(DIM_TITLES[d], fontsize=8)
        _style_ax(ax, "")
    axes[0].legend(fontsize=6.5, ncol=2, loc="upper right")
    axes[-1].set_xlabel("tick (30 Hz)")
    fig.suptitle(f"Full-episode commanded vs GT (episode {ep}, lag={run0['lag_ticks']} ticks)", fontsize=10)
    fig.savefig(out / "03_trajectory_overview.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("[fig] 03_trajectory_overview.png")


def plot_jitter_drift(runs, args, out):
    ep = args.detail_episode
    names = [nm for nm in STRATEGY_LABELS if (nm, ep) in runs]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for nm in names:
        run = runs[(nm, ep)]
        cmd = run["commanded"]
        steps = np.linalg.norm(np.diff(cmd[:, :3], axis=0), axis=1) * 1000
        axes[0].plot(
            np.arange(1, len(steps) + 1),
            steps,
            "-",
            lw=1.0,
            color=STRATEGY_COLORS[nm],
            alpha=0.8,
            label=STRATEGY_LABELS[nm],
        )
        drift = [ee_xyz_mm(cmd[t], run["gt"][t]) for t in range(min(len(cmd), len(run["gt"])))]
        axes[1].plot(
            np.arange(len(drift)),
            drift,
            "-",
            lw=1.1,
            color=STRATEGY_COLORS[nm],
            alpha=0.85,
            label=STRATEGY_LABELS[nm],
        )
    axes[0].set_ylabel("EE step [mm/tick]")
    axes[1].set_ylabel("drift vs GT [mm]")
    axes[1].set_xlabel("tick (30 Hz)")
    for s in runs[(names[0], ep)]["chunk_starts"][1:]:
        for ax in axes:
            ax.axvline(s, color="tab:red", ls=":", lw=0.6, alpha=0.5)
    axes[0].legend(fontsize=6.5, ncol=2)
    for ax in axes:
        _style_ax(ax, "")
    fig.suptitle(f"Per-tick jitter (step) & drift (episode {ep})", fontsize=10)
    fig.savefig(out / "04_jitter_drift.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("[fig] 04_jitter_drift.png")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--episodes", type=int, nargs="+", default=[0])
    p.add_argument(
        "--ee_key",
        default="action",
        help="Dataset column with the 7D absolute EE pose (UMI relative-EE: 'action').",
    )
    p.add_argument("--detail_episode", type=int, default=0)
    p.add_argument("--n_action_steps", type=int, default=30)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--replan_every",
        type=int,
        default=30,
        help="SYNC/UMI re-plan cadence in ticks (= n_action_steps for sync).",
    )
    p.add_argument(
        "--lag_ticks",
        type=int,
        nargs="+",
        default=[3],
        help="Tracking lag(s) to sweep: measured pose = commanded delayed by N ticks (>=1).",
    )
    p.add_argument(
        "--detail_lag",
        type=int,
        default=None,
        help="Lag to use for the detail/overview figures (default: the largest swept lag).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument(
        "--waypoint_stride",
        type=int,
        default=1,
        help="Subsample chunk to every Nth waypoint (interpolation-as-lowpass test).",
    )
    p.add_argument("--out", required=True)
    return p.parse_args()


def _build_strategies(args):
    return [
        ("sync_replay", SyncReplay(replan_every=args.replan_every)),
        (
            "umi_splice",
            UmiInterp(
                anchor="measured", replan_every=args.replan_every, waypoint_stride=args.waypoint_stride
            ),
        ),
        (
            "cmd_reanchor",
            UmiInterp(
                anchor="commanded", replan_every=args.replan_every, waypoint_stride=args.waypoint_stride
            ),
        ),
        (
            "async_ensemble",
            AsyncEnsemble(replan_every=min(10, args.replan_every), chunk_len=args.n_action_steps),
        ),
    ]


def plot_lag_sweep(runs_by_lag, args, out):
    """Boundary velocity change and drift vs tracking lag, per strategy."""
    lags = sorted(runs_by_lag)
    names = [nm for nm in STRATEGY_LABELS if any((nm, ep) in runs_by_lag[lags[0]] for ep in args.episodes)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for nm in names:
        boundary_y, drift_y = [], []
        for lag in lags:
            sums = [
                summarize(runs_by_lag[lag][(nm, ep)]) for ep in args.episodes if (nm, ep) in runs_by_lag[lag]
            ]
            boundary_y.append(
                np.mean([s["boundary_velocity_change_mean_mm"] for s in sums]) if sums else np.nan
            )
            drift_y.append(np.mean([s["drift_mean_mm"] for s in sums]) if sums else np.nan)
        axes[0].plot(
            lags, boundary_y, "-o", color=STRATEGY_COLORS[nm], lw=1.8, ms=5, label=STRATEGY_LABELS[nm]
        )
        axes[1].plot(lags, drift_y, "-o", color=STRATEGY_COLORS[nm], lw=1.8, ms=5, label=STRATEGY_LABELS[nm])
    axes[0].set_xlabel("tracking lag [ticks]")
    axes[0].set_ylabel("boundary velocity change [mm/tick]")
    axes[0].set_title("Chunk-boundary velocity discontinuity vs lag")
    axes[1].set_xlabel("tracking lag [ticks]")
    axes[1].set_ylabel("drift vs GT  [mm]")
    axes[1].set_title("Executed-command drift vs GT")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6.5)
    fig.suptitle("Corrected execution study: boundary smoothness and drift across tracking lag", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "05_lag_sweep.png", dpi=150)
    plt.close(fig)
    print("[fig] 05_lag_sweep.png")


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    lags = sorted({max(1, int(x)) for x in args.lag_ticks})
    detail_lag = args.detail_lag if args.detail_lag is not None else max(lags)

    print(f"Loading policy from {args.pretrained_path}")
    policy, preprocessor, postprocessor = load_policy_and_processors(args.pretrained_path, device)
    action_indices = policy.config.action_delta_indices
    if action_indices is None or action_indices[:2] != [-1, 0]:
        raise ValueError(
            "This experiment's timestamp alignment requires UMI action indices to begin "
            f"[-1, 0], got {action_indices}"
        )

    prediction_every = min(10, args.replan_every)
    shared_by_episode = {}
    for ep in args.episodes:
        print(f"  episode {ep} / fixed GT policy predictions ...", flush=True)
        shared_by_episode[ep] = collect_shared_predictions(
            args, ep, policy, preprocessor, postprocessor, device, prediction_every
        )

    runs_by_lag: dict[int, dict] = {}
    for lag in lags:
        runs_by_lag[lag] = {}
        for ep in args.episodes:
            for name, strat in _build_strategies(args):
                print(f"  lag={lag} episode {ep} / strategy {name} ...", flush=True)
                run = run_strategy(args, shared_by_episode[ep], strat, lag)
                runs_by_lag[lag][(name, ep)] = run

    # ---- metrics (per lag, aggregated across episodes) ----
    all_metrics = {}
    for lag in lags:
        agg = {}
        for name in STRATEGY_LABELS:
            sums = [
                summarize(runs_by_lag[lag][(name, ep)])
                for ep in args.episodes
                if (name, ep) in runs_by_lag[lag]
            ]
            if not sums:
                continue
            agg[name] = {
                "episodes": len(sums),
                "drift_mean_mm": float(np.mean([s["drift_mean_mm"] for s in sums])),
                "drift_max_mm": float(np.mean([s["drift_max_mm"] for s in sums])),
                "reversal_rate": float(np.mean([s["reversal_rate"] for s in sums])),
                "mean_step_mm": float(np.mean([s["mean_step_mm"] for s in sums])),
                "second_difference_m_per_tick2": float(
                    np.mean([s["second_difference_m_per_tick2"] for s in sums])
                ),
                "tracking_gap_mean_mm": float(np.mean([s["tracking_gap_mean_mm"] for s in sums])),
                "tracking_gap_max_mm": float(np.mean([s["tracking_gap_max_mm"] for s in sums])),
                "replan_anchor_gap_mean_mm": float(np.mean([s["replan_anchor_gap_mean_mm"] for s in sums])),
                "replan_anchor_gap_max_mm": float(np.mean([s["replan_anchor_gap_max_mm"] for s in sums])),
                "boundary_step_mean_mm": float(np.mean([s["boundary_step_mean_mm"] for s in sums])),
                "boundary_velocity_change_mean_mm": float(
                    np.mean([s["boundary_velocity_change_mean_mm"] for s in sums])
                ),
                "total_switches": int(sum(s["n_switches"] for s in sums)),
            }
        all_metrics[f"lag_{lag}"] = agg
    with open(out / "metrics.json", "w") as f:
        json.dump(
            {
                "aggregate_per_lag": all_metrics,
                "config": {
                    "lag_ticks": lags,
                    "replan_every": args.replan_every,
                    "n_action_steps": args.n_action_steps,
                    "fps": args.fps,
                    "waypoint_stride": args.waypoint_stride,
                    "episodes": args.episodes,
                },
            },
            f,
            indent=2,
        )

    print(f"\n=== Aggregate metrics @ lag={detail_lag} ticks (stride={args.waypoint_stride}) ===")
    hdr = (
        f"{'strategy':<16}{'drift':>8}{'gap':>8}{'rev%':>7}{'step':>8}"
        f"{'d2':>9}{'bnd_dv':>9}{'bnd_step':>10}{'#sw':>5}"
    )
    print(hdr)
    for name, a in all_metrics[f"lag_{detail_lag}"].items():
        print(
            f"{name:<16}{a['drift_mean_mm']:>8.2f}{a['tracking_gap_mean_mm']:>8.2f}"
            f"{a['reversal_rate'] * 100:>7.1f}{a['mean_step_mm']:>8.2f}"
            f"{a['second_difference_m_per_tick2'] * 1000:>9.3f}"
            f"{a['boundary_velocity_change_mean_mm']:>9.2f}"
            f"{a['boundary_step_mean_mm']:>10.2f}{a['total_switches']:>5}"
        )

    # ---- figures (detail set at the first lag; sweep across all lags) ----
    det = runs_by_lag[detail_lag]
    plot_two_chunk_detail(det, args, out)
    plot_switch_jumps(det, out)
    plot_trajectory_overview(det, args, out)
    plot_jitter_drift(det, args, out)
    if len(lags) > 1:
        plot_lag_sweep(runs_by_lag, args, out)
    print("\noutputs in", out)


if __name__ == "__main__":
    main()

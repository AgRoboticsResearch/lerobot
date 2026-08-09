#!/usr/bin/env python
"""Detailed episode-level diagnostics for one mock or deploy control log.

The plots intentionally keep the complete episode timeline while separating:
  - logged robot state (``current_ee``),
  - optional dataset reference pose aligned by mock control tick,
  - executed/aggregated absolute command (``action_agg`` / ``action_ee``),
  - the incoming pre-ensemble target (``action_abs``), and
  - chunk arrivals / queue / response latency.

Usage::

    uv run python examples/umi_relative_ee/control_debug/make_episode_debug_figs.py \
        logs/mock_20260808_080851

The output is written beside the log under ``episode_debug/`` unless ``--out``
is supplied.  It works with old logs too; unavailable provenance fields are
shown as gaps rather than fabricated.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EE_LABELS = ("x [m]", "y [m]", "z [m]", "rot_x [rad]", "rot_y [rad]", "rot_z [rad]", "gripper")
COLORS = {
    "gt": "#008300",
    "cmd": "#2a78d6",
    "incoming": "#eb6834",
    "arrival": "#e34948",
    "queue": "#4a3aa7",
    "error": "#e87ba4",
    "latency": "#1baf7a",
    "muted": "#777777",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detailed one-episode control-log plots")
    p.add_argument("stem", help="Log stem, e.g. logs/mock_20260808_080851")
    p.add_argument("--out", default=None, help="Output directory (default: <log_dir>/episode_debug/<stem>)")
    p.add_argument("--dataset", default=None, help="Optional LeRobotDataset for action_timestep-aligned GT")
    p.add_argument("--episode", type=int, default=0, help="Dataset episode used for aligned GT (default: 0)")
    p.add_argument("--max_chunk_panels", type=int, default=8, help="Max chunk detail panels in figure 8")
    return p.parse_args()


def vec(value: str | None) -> np.ndarray | None:
    if value is None or value == "":
        return None
    return np.fromstring(value, sep=";")


def load(stem: Path) -> tuple[dict[str, np.ndarray], list[dict[str, str]], list[dict[str, str]], dict]:
    d = {k: v for k, v in np.load(stem.with_suffix(".npz"), allow_pickle=False).items()}
    with stem.with_suffix(".csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    merge_path = Path(f"{stem}_merge.csv")
    merges: list[dict[str, str]] = []
    if merge_path.exists():
        with merge_path.open(newline="") as fh:
            merges = list(csv.DictReader(fh))
    meta = json.loads(stem.with_suffix(".json").read_text())
    if len(rows) != len(d["t_s"]):
        raise ValueError(f"CSV/NPZ length mismatch: {len(rows)} vs {len(d['t_s'])}")
    d["state"] = np.array([r.get("state", "") for r in rows])
    d["skip_reason"] = np.array([r.get("skip_reason", "") for r in rows])
    return d, rows, merges, meta


def valid_data(d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    valid = d["ik_ok"].astype(bool)
    # For mock logs action_ee/action_agg are equivalent; prefer action_agg when present.
    command = d.get("action_agg", d["action_ee"])
    if command.ndim != 2 or command.shape[1] != 7:
        command = d["action_ee"]
    return valid, command


def gt_at_control_ticks(d: dict[str, np.ndarray], dataset_gt: np.ndarray | None) -> np.ndarray | None:
    """Align mock dataset frames with logged control ticks.

    ``mock_async_client`` loops over dataset frame ``t`` while logging tick ``t``.
    This is separate from ``action_timestep``: the latter is the logical queued
    command horizon, whereas this reference is the dataset frame shown as the
    image at the control tick.
    """
    if dataset_gt is None:
        return None
    out = np.full_like(d["current_ee"], np.nan)
    tick = np.rint(d["tick"]).astype(int)
    ok = np.isfinite(d["tick"]) & (tick >= 0) & (tick < len(dataset_gt))
    out[ok] = dataset_gt[tick[ok]]
    return out


def arrival_indices(d: dict[str, np.ndarray], valid: np.ndarray) -> np.ndarray:
    ids = d.get("chunk_id")
    if ids is None:
        return np.array([], dtype=int)
    finite = valid & np.isfinite(ids)
    idx = np.flatnonzero(finite)
    if idx.size == 0:
        return idx
    return idx[np.r_[True, ids[idx[1:]] != ids[idx[:-1]]]]


def decorate(ax: plt.Axes, t: np.ndarray, arrivals: np.ndarray) -> None:
    for i in arrivals:
        ax.axvline(t[i], color=COLORS["arrival"], alpha=0.18, lw=0.8, zorder=0)
    ax.grid(True, alpha=0.18)
    ax.set_xlim(float(t[0]), float(t[-1]))


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _aa_to_mat(pose7):
    """7D [xyz, axis-angle(3), gripper] → 4x4 (gripper ignored)."""
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, 3] = pose7[:3]
    T[:3, :3] = Rotation.from_rotvec(np.asarray(pose7[3:6])).as_matrix()
    return T


def predicted_from_gt_anchor(d: dict, valid: np.ndarray):
    """Re-anchor the executed absolute targets at the GT pose at each chunk's start.

    The relative prediction is extracted from (action_agg, chunk_ref_ee), then
    recomposed with current_ee (= GT in mock) at the chunk's first tick. Shows what
    the policy would predict if anchored at GT instead of the drifted pose.
    Returns (n,7) with NaN on invalid ticks, or None if chunk_ref_ee is absent.
    """
    from scipy.spatial.transform import Rotation
    cmd = d.get("action_agg", d.get("action_ee"))
    ref = d.get("chunk_ref_ee")
    state = d["current_ee"]
    cid = d.get("chunk_id")
    if cmd is None or ref is None or cid is None:
        return None
    n = len(state)
    out = np.full((n, 7), np.nan)
    chunks: dict = {}
    for i in range(n):
        if valid[i]:
            chunks.setdefault(int(cid[i]), []).append(i)
    for ticks in chunks.values():
        first = ticks[0]
        gt_start = state[first]
        anchor = ref[first]
        T_anchor_inv = np.linalg.inv(_aa_to_mat(anchor))
        T_gt = _aa_to_mat(gt_start)
        for i in ticks:
            T_rel = T_anchor_inv @ _aa_to_mat(cmd[i])
            T_new = T_gt @ T_rel
            out[i, :3] = T_new[:3, 3]
            out[i, 3:6] = Rotation.from_matrix(T_new[:3, :3]).as_rotvec()
            out[i, 6] = cmd[i, 6]
    return out


def fig_01_overview(
    d: dict[str, np.ndarray], out: Path, arrivals: np.ndarray,
    reference_gt: np.ndarray | None = None,
) -> None:
    t = d["t_s"]
    valid, command = valid_data(d)
    state = d["current_ee"]
    pred_gt = predicted_from_gt_anchor(d, valid)
    fig, axes = plt.subplots(7, 1, figsize=(13, 15), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t, state[:, j], color=COLORS["gt"], lw=1.0, label="current_ee (GT in mock, robot state in deploy)")
        if reference_gt is not None and not np.allclose(reference_gt, state, atol=1e-6, equal_nan=True):
            ax.plot(t, reference_gt[:, j], color=COLORS["muted"], ls="--", lw=0.8,
                    label="dataset GT at control tick")
        y = np.where(valid, command[:, j], np.nan)
        ax.plot(t, y, color=COLORS["cmd"], lw=1.1, marker=".", ms=2.0, label="executed command (from actual anchor)")
        if pred_gt is not None:
            ax.plot(t, pred_gt[:, j], color=COLORS["incoming"], ls="--", lw=0.9,
                    label="predicted from GT anchor")
        decorate(ax, t, arrivals)
        ax.set_ylabel(EE_LABELS[j], fontsize=8)
        if j == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=2)
        if j == 6:
            ax.set_xlabel("episode time [s] — red lines = chunk arrivals")
    fig.suptitle("Episode overview — robot state, dataset GT, and executed absolute EE command", fontsize=13)
    save(fig, out, "01_episode_overview_gt_vs_command.png")


def fig_02_errors(
    d: dict[str, np.ndarray], out: Path, arrivals: np.ndarray,
    reference_gt: np.ndarray | None = None,
) -> None:
    t = d["t_s"]
    valid, command = valid_data(d)
    gt = reference_gt if reference_gt is not None else d["current_ee"]
    err = np.where(valid[:, None], command - gt, np.nan)
    state_err = d["current_ee"] - gt
    units = np.array([1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0, 1.0])
    labels = ("x error [mm]", "y error [mm]", "z error [mm]", "rot_x error [rad]",
              "rot_y error [rad]", "rot_z error [rad]", "gripper error")
    fig, axes = plt.subplots(7, 1, figsize=(13, 15), sharex=True)
    for j, ax in enumerate(axes):
        ax.axhline(0, color=COLORS["muted"], lw=0.7)
        target_label = "executed command − dataset GT" if reference_gt is not None else "executed command − robot state"
        ax.plot(t, err[:, j] * units[j], color=COLORS["error"], lw=1.0, marker=".", ms=2,
                label=target_label)
        if reference_gt is not None:
            ax.plot(t, state_err[:, j] * units[j], color=COLORS["gt"], ls="--", lw=0.8,
                    label="robot state − dataset GT")
        decorate(ax, t, arrivals)
        ax.set_ylabel(labels[j], fontsize=8)
        if j == 0 and reference_gt is not None:
            ax.legend(fontsize=7)
        if j == 6:
            ax.set_xlabel("episode time [s]")
    target_name = "dataset GT" if reference_gt is not None else "logged robot state"
    fig.suptitle(f"Episode error — executed command minus {target_name}", fontsize=13)
    save(fig, out, "02_episode_command_minus_gt_error.png")


def fig_03_increments(
    d: dict[str, np.ndarray], out: Path, arrivals: np.ndarray,
    reference_gt: np.ndarray | None = None,
) -> None:
    t = d["t_s"]
    valid, command = valid_data(d)
    gt = reference_gt if reference_gt is not None else d["current_ee"]
    cmd_step = np.linalg.norm(np.diff(command[:, :3], axis=0), axis=1) * 1000.0
    gt_step = np.linalg.norm(np.diff(gt[:, :3], axis=0), axis=1) * 1000.0
    state_step = np.linalg.norm(np.diff(d["current_ee"][:, :3], axis=0), axis=1) * 1000.0
    signed = np.diff(command[:, :3], axis=0)
    direction = np.nanmean(signed[valid[1:] & valid[:-1]], axis=0)
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    signed_step = np.sum(signed * direction, axis=1) * 1000.0
    valid_step = valid[1:] & valid[:-1]
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(t[1:], np.where(valid_step, cmd_step, np.nan), color=COLORS["cmd"], marker=".", ms=2, lw=0.8, label="command step")
    axes[0].plot(t[1:], np.where(valid_step, gt_step, np.nan), color=COLORS["gt"], lw=0.9, label="dataset GT step")
    if reference_gt is not None:
        axes[0].plot(t[1:], state_step, color=COLORS["muted"], ls="--", lw=0.8,
                     label="robot state step")
    axes[0].set_ylabel("XYZ step [mm]")
    axes[0].legend(fontsize=8)
    axes[1].plot(t[1:], np.where(valid_step, signed_step, np.nan), color=COLORS["error"], marker=".", ms=2, lw=0.8)
    axes[1].axhline(0, color=COLORS["muted"], lw=0.7)
    axes[1].set_ylabel("signed command step [mm]")
    axes[2].plot(t, d["queue"], color=COLORS["queue"], lw=1.0, drawstyle="steps-post")
    axes[2].set_ylabel("queue")
    axes[2].set_xlabel("episode time [s]")
    for ax in axes:
        decorate(ax, t, arrivals)
    fig.suptitle("Episode increments — motion magnitude, direction reversals, and queue", fontsize=13)
    save(fig, out, "03_episode_steps_direction_queue.png")


def fig_04_timing(d: dict[str, np.ndarray], out: Path, arrivals: np.ndarray) -> None:
    t = d["t_s"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(t, d["tick_dt_ms"], color=COLORS["cmd"], marker=".", ms=2, lw=0.8)
    axes[0].axhline(1000.0 / 30.0, color=COLORS["muted"], ls="--", lw=0.8, label="33.3 ms target")
    axes[0].set_ylabel("tick dt [ms]")
    axes[0].set_ylim(0, min(100.0, max(100.0, float(np.nanpercentile(d["tick_dt_ms"], 99.5)))))
    axes[0].legend(fontsize=8)
    axes[1].plot(t, d["work_ms"], color=COLORS["incoming"], marker=".", ms=2, lw=0.8)
    axes[1].set_ylabel("work [ms]")
    axes[1].set_ylim(0, min(100.0, max(100.0, float(np.nanpercentile(d["work_ms"], 99.5)))))
    finite = np.isfinite(d["e2e_ms"])
    axes[2].scatter(t[finite], d["e2e_ms"][finite], color=COLORS["latency"], s=14)
    axes[2].set_ylabel("e2e [ms]")
    axes[2].set_xlabel("episode time [s]")
    for ax in axes:
        decorate(ax, t, arrivals)
    fig.suptitle("Episode timing — response arrivals are marked in red", fontsize=13)
    save(fig, out, "04_episode_timing_latency.png")


def fig_05_provenance(
    d: dict[str, np.ndarray], out: Path, arrivals: np.ndarray,
    reference_gt: np.ndarray | None = None,
) -> None:
    t = d["t_s"]
    valid, command = valid_data(d)
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    chunk = d.get("chunk_id", np.full(len(t), np.nan))
    axes[0].plot(t, np.where(valid, chunk, np.nan), color=COLORS["queue"], marker=".", ms=2, lw=0.8)
    axes[0].set_ylabel("chunk id")
    for label, key, color in (("aggregated executed", "action_agg", COLORS["cmd"]),
                              ("incoming pre-ensemble", "action_abs", COLORS["incoming"])):
        arr = d.get(key)
        if arr is None or arr.ndim != 2 or arr.shape[1] < 3:
            continue
        axes[1].plot(t, np.where(valid, arr[:, 0] * 1000.0, np.nan), color=color, lw=0.9, marker=".", ms=1.7, label=label)
    axes[1].plot(t, d["current_ee"][:, 0] * 1000.0, color=COLORS["gt"], lw=0.9,
                 label="mock robot state / current_ee")
    if reference_gt is not None:
        axes[1].plot(t, reference_gt[:, 0] * 1000.0, color=COLORS["muted"], ls="--", lw=0.8,
                     label="dataset GT at control tick")
    axes[1].set_ylabel("x target [mm]")
    axes[1].legend(fontsize=8)
    ref = d.get("chunk_ref_ee")
    if ref is not None and ref.ndim == 2 and ref.shape[1] >= 3:
        axes[2].plot(t, np.where(valid, (command[:, 0] - ref[:, 0]) * 1000.0, np.nan), color=COLORS["error"], lw=0.9, marker=".", ms=1.7)
    axes[2].axhline(0, color=COLORS["muted"], lw=0.7)
    axes[2].set_ylabel("command-anchor x [mm]")
    axes[3].plot(t, d["queue"], color=COLORS["queue"], lw=1.0, drawstyle="steps-post")
    axes[3].set_ylabel("queue")
    axes[3].set_xlabel("episode time [s]")
    for ax in axes:
        decorate(ax, t, arrivals)
    fig.suptitle("Chunk provenance — robot state, GT, incoming target, aggregate, and anchor", fontsize=13)
    save(fig, out, "05_episode_chunk_provenance.png")


def fig_06_merges(merges: list[dict[str, str]], out: Path) -> None:
    if not merges:
        return
    ts = np.array([float(r["timestep"]) for r in merges])
    w = np.array([float(r["weight"]) for r in merges])
    def norm_diff(a: str, b: str) -> np.ndarray:
        va = np.array([vec(r[a]) for r in merges])
        vb = np.array([vec(r[b]) for r in merges])
        return np.linalg.norm(va[:, :3] - vb[:, :3], axis=1) * 1000.0
    old_new = norm_diff("existing_abs", "incoming_abs")
    old_agg = norm_diff("existing_abs", "aggregated_abs")
    new_agg = norm_diff("incoming_abs", "aggregated_abs")
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(ts, w, color=COLORS["queue"], marker=".", ms=2, lw=0.8)
    axes[0].set_ylabel("old-target weight")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].plot(ts, old_new, color=COLORS["incoming"], marker=".", ms=2, lw=0.8, label="|existing − incoming|")
    axes[1].set_ylabel("blend disagreement [mm]")
    axes[1].legend(fontsize=8)
    axes[2].plot(ts, old_agg, color=COLORS["cmd"], marker=".", ms=2, lw=0.8, label="existing → aggregate")
    axes[2].plot(ts, new_agg, color=COLORS["gt"], marker=".", ms=2, lw=0.8, label="incoming → aggregate")
    axes[2].set_ylabel("blend movement [mm]")
    axes[2].set_xlabel("action timestep")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.18)
    fig.suptitle("Temporal-ensemble merge audit — one dot per overlapping action", fontsize=13)
    save(fig, out, "06_episode_merge_audit.png")


def fig_07_chunk_windows(
    d: dict[str, np.ndarray], out: Path, arrivals: np.ndarray, max_panels: int,
    reference_gt: np.ndarray | None = None,
) -> None:
    valid, command = valid_data(d)
    ids = d.get("chunk_id")
    if ids is None:
        return
    chunks = [int(x) for x in ids[arrivals] if np.isfinite(x)]
    chunks = chunks[:max_panels]
    if not chunks:
        return
    fig, axes = plt.subplots(len(chunks), 1, figsize=(12, max(3, 2.0 * len(chunks))), squeeze=False)
    t = d["t_s"]
    for ax, cid in zip(axes[:, 0], chunks):
        m = valid & (ids == cid)
        ax.plot(t[m], command[m, 0] * 1000.0, color=COLORS["cmd"], marker=".", ms=2, lw=0.9)
        ax.plot(t[m], d["current_ee"][m, 0] * 1000.0, color=COLORS["gt"], lw=0.8,
                label="robot state x")
        if reference_gt is not None:
            ax.plot(t[m], reference_gt[m, 0] * 1000.0, color=COLORS["muted"], ls="--", lw=0.8,
                    label="dataset GT x")
        ax.set_ylabel(f"c{cid}\nx [mm]", fontsize=8)
        ax.grid(True, alpha=0.18)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("episode time [s]")
    fig.suptitle("First chunk windows — command, robot state, and dataset GT x", fontsize=13)
    save(fig, out, "07_episode_chunk_windows.png")


def fig_08_aligned_error(
    d: dict[str, np.ndarray], out: Path, dataset_gt: np.ndarray, arrivals: np.ndarray
) -> None:
    """Separate prediction error from current-tick/horizon and ensemble error."""
    t = d["t_s"]
    valid, command = valid_data(d)
    ts = d["action_timestep"].astype(float)
    aligned = valid & np.isfinite(ts) & (ts >= 0) & (ts < len(dataset_gt))
    idx = np.flatnonzero(aligned)
    if idx.size == 0:
        return
    gt_target = dataset_gt[ts[idx].astype(int)]
    raw = d.get("action_abs", np.full_like(command, np.nan))[idx]
    agg = command[idx]
    current = d["current_ee"][idx]
    # XYZ in mm; rotation/gripper in native units.
    raw_xyz = np.linalg.norm(raw[:, :3] - gt_target[:, :3], axis=1) * 1000.0
    agg_xyz = np.linalg.norm(agg[:, :3] - gt_target[:, :3], axis=1) * 1000.0
    current_xyz = np.linalg.norm(agg[:, :3] - current[:, :3], axis=1) * 1000.0
    state_gt_xyz = np.linalg.norm(current[:, :3] - gt_target[:, :3], axis=1) * 1000.0
    anchor = d.get("chunk_ref_ee", np.full_like(command, np.nan))[idx]
    raw_anchor_xyz = np.linalg.norm(raw[:, :3] - anchor[:, :3], axis=1) * 1000.0
    raw_dim = np.abs(raw - gt_target)
    agg_dim = np.abs(agg - gt_target)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(t[idx], raw_xyz, color=COLORS["incoming"], marker=".", ms=2, lw=0.8, label="raw incoming vs GT[action_timestep]")
    axes[0].plot(t[idx], raw_anchor_xyz, color=COLORS["gt"], marker=".", ms=2, lw=0.8, label="raw incoming vs chunk anchor")
    axes[0].plot(t[idx], agg_xyz, color=COLORS["cmd"], marker=".", ms=2, lw=0.8, label="aggregate vs GT[action_timestep]")
    axes[0].plot(t[idx], state_gt_xyz, color=COLORS["muted"], marker=".", ms=2, lw=0.8,
                 label="robot state vs GT[action_timestep]")
    axes[0].plot(t[idx], current_xyz, color=COLORS["error"], marker=".", ms=2, lw=0.8,
                 label="aggregate vs current robot state")
    axes[0].set_ylabel("XYZ error [mm]")
    axes[0].legend(fontsize=8)
    axes[1].plot(t[idx], raw_dim[:, 3], color=COLORS["incoming"], lw=0.8, label="raw rot_x")
    axes[1].plot(t[idx], agg_dim[:, 3], color=COLORS["cmd"], lw=0.8, label="aggregate rot_x")
    axes[1].plot(t[idx], raw_dim[:, 4], color=COLORS["incoming"], ls="--", lw=0.8, label="raw rot_y")
    axes[1].plot(t[idx], agg_dim[:, 4], color=COLORS["cmd"], ls="--", lw=0.8, label="aggregate rot_y")
    axes[1].plot(t[idx], raw_dim[:, 5], color=COLORS["incoming"], ls=":", lw=0.9, label="raw rot_z")
    axes[1].plot(t[idx], agg_dim[:, 5], color=COLORS["cmd"], ls=":", lw=0.9, label="aggregate rot_z")
    axes[1].set_ylabel("rotation abs error [rad]")
    axes[1].legend(fontsize=7, ncol=3)
    axes[2].plot(t[idx], raw_dim[:, 6], color=COLORS["incoming"], lw=0.8, label="raw gripper")
    axes[2].plot(t[idx], agg_dim[:, 6], color=COLORS["cmd"], lw=0.8, label="aggregate gripper")
    axes[2].set_ylabel("gripper abs error")
    axes[2].set_xlabel("episode time [s]")
    axes[2].legend(fontsize=8)
    for ax in axes:
        decorate(ax, t, arrivals)
    fig.suptitle("Aligned error decomposition — target, robot state, and GT at action_timestep", fontsize=13)
    save(fig, out, "08_episode_aligned_model_vs_ensemble_error.png")


def load_dataset_gt(dataset_path: str, episode: int) -> np.ndarray:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(dataset_path, episodes=[episode])
    rows = []
    for i in range(len(ds)):
        frame = ds[i]
        for key in ("observation.ee", "observation.state", "action"):
            if key in frame:
                rows.append(_state_to_ee7(frame[key]))
                break
        else:
            raise KeyError("Dataset frame has none of observation.ee, observation.state, action")
    return np.asarray(rows, dtype=np.float32)


def _state_to_ee7(state) -> np.ndarray:
    a = np.asarray(state.cpu() if hasattr(state, "cpu") else state, dtype=np.float32).reshape(-1)
    if a.size == 7:
        return a
    if a.size == 14:
        return a[-7:]
    raise ValueError(f"Expected 7D or 14D EE state, got {a.shape}")


def main() -> None:
    args = parse_args()
    stem = Path(args.stem)
    if stem.suffix:
        stem = stem.with_suffix("")
    if args.out:
        out = Path(args.out)
    else:
        out = stem.parent / "episode_debug" / stem.name
    out.mkdir(parents=True, exist_ok=True)
    d, _rows, merges, meta = load(stem)
    dataset_gt = load_dataset_gt(args.dataset, args.episode) if args.dataset else None
    reference_gt = gt_at_control_ticks(d, dataset_gt)
    valid, _command = valid_data(d)
    arrivals = arrival_indices(d, valid)
    fig_01_overview(d, out, arrivals, reference_gt)
    fig_02_errors(d, out, arrivals, reference_gt)
    fig_03_increments(d, out, arrivals, reference_gt)
    fig_04_timing(d, out, arrivals)
    fig_05_provenance(d, out, arrivals, reference_gt)
    fig_06_merges(merges, out)
    fig_07_chunk_windows(d, out, arrivals, args.max_chunk_panels, reference_gt)
    if dataset_gt is not None:
        fig_08_aligned_error(d, out, dataset_gt, arrivals)
    stats = {
        "source": str(stem),
        "metadata": meta,
        "n_ticks": int(len(d["t_s"])),
        "n_valid": int(valid.sum()),
        "n_chunk_arrivals": int(len(arrivals)),
        "n_merge_events": int(len(merges)),
        "aligned_gt": bool(args.dataset),
        "dataset": args.dataset,
        "episode": args.episode if args.dataset else None,
        "valid_fraction": float(valid.mean()),
        "figure_dir": str(out),
    }
    (out / "episode_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

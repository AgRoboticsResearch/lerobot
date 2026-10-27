"""Shared per-tick control logger for the UMI relative-EE Piper deploy scripts.

Both the sync deploy (``deploy_umi_relative_ee_piper.py``) and the async client
(``async_umi_relative_ee_piper_client.py``) import :class:`ControlLogger` /
:func:`make_control_logger` from here so their ``--log`` output shares ONE
schema (same CSV columns, same JSON meta keys) and runs from either script can
be diffed directly.

Design note (2026-10-27): the logger appends one CSV row per tick and flushes
to the OS — O(1) per tick, crash-safe against SIGKILL. It never rewrites files
mid-run: the previous design buffered all rows in memory and rebuilt the full
csv+npz+json every 150 ticks on the control thread, which stalled the loop for
seconds (growing with session length) — see the 2026-10-20 deploy-log
forensics. The npz sibling was dropped entirely: it held the same numbers as
the CSV and had no consumer (regenerate from CSV on demand via pandas).
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import time
from pathlib import Path

import numpy as np

from lerobot.datasets.compute_stats import RunningQuantileStats

logger = logging.getLogger(__name__)

# Scalars that get a streaming distribution summary in the JSON at close.
_STAT_FIELDS: tuple[str, ...] = (
    "tick_dt_ms", "work_ms", "e2e_ms", "ee_delta_m", "joint_delta_max_rad",
)


class ControlLogger:
    """Per-tick control log for offline stutter / timing analysis (``--log``).

    Off by default; zero overhead when disabled. Each loop tick records the
    timeline, the FULL tick period (work + pacing sleep — so the reported Hz is
    honest rather than the work-only time), loop state, queue depth, the popped
    absolute-EE action (the "trajectory sent to the robot"), the IK joint command
    ("real joint commands"), read-back joints, IK skip events, and inference
    latency snapshots.

    Files:
      <stem>.csv   one row per tick (scalars + ';'-joined vectors) — appended and
                   flushed every tick (O(1); durable against SIGKILL)
      <stem>.json  run metadata — written once at start; the summary block
                   (tick counts + per-field distributions) is added at close

    Per-tick UMI relative-EE chunk-provenance fields (added so the temporal
    ensemble can be audited offline — see log_merge for the per-timestep blend):
      chunk_id        which action chunk this tick's target came from
      chunk_ref_ee    7D EE pose that anchored/conditioned that chunk (T_anchor)
      action_abs      7D absolute EE target from that chunk (T_anchor ∘ ΔT), pre-ensemble
      action_agg      7D aggregated absolute target actually executed (post weighted_average)
      action_rel      raw relative action ΔT (model output; SYNC: in-process, ASYNC: server-reported)

    And, for async runs only, a third sibling records each ensemble blend:
      <stem>_merge.csv  one row per (chunk × overlapping timestep): the existing
                        absolute target, the incoming chunk's absolute target +
                        anchor, the weight, and the resulting aggregated target —
                        so you can confirm the ensemble averages ABSOLUTE targets
                        (same base frame), not relative ΔT's directly.
    """

    SCALAR_FIELDS: tuple[str, ...] = (
        "step", "tick", "t_s", "tick_dt_ms", "work_ms", "state", "queue",
        "popped", "ik_ok", "skip_reason", "action_timestep", "chunk_id",
        "ee_delta_m", "joint_delta_max_rad", "gripper", "e2e_ms", "wire_ms", "server_ms",
    )
    ARRAY_FIELDS: tuple[str, ...] = (
        "action_ee", "current_ee", "ik_joints_rad", "current_joints_rad",
        "chunk_ref_ee", "action_abs", "action_agg", "action_rel",
    )
    # Per-blend columns for the async temporal-ensemble audit log (<stem>_merge.csv).
    MERGE_FIELDS: tuple[str, ...] = (
        "step", "tick", "timestep", "chunk_id", "weight",
        "existing_abs", "incoming_abs", "aggregated_abs", "ref_ee",
    )

    def __init__(self, stem: str, *, fps: int, meta: dict, flush_every: int = 150) -> None:
        # ``flush_every`` is accepted for call compatibility but IGNORED: the CSV
        # is appended and flushed every tick, so there is no periodic rewrite to
        # schedule (the old every-150-ticks full rewrite froze the control loop).
        del flush_every
        self.stem = stem
        self.fps = fps
        self.meta = dict(meta)
        self.meta["fps_target"] = fps
        self._loop_t0: float | None = None
        self._fh: csv.DictWriter | None = None
        self._fh_file = None
        self._merge_writer: csv.DictWriter | None = None
        self._merge_file = None
        self._counts = {
            "n_ticks": 0, "n_popped": 0, "n_ik_ok": 0,
            "n_ik_skip": 0, "n_underrun": 0, "n_paused": 0,
        }
        self._stat_n = {name: 0 for name in _STAT_FIELDS}
        self._stats = {
            name: RunningQuantileStats(quantile_list=[0.50, 0.95, 0.99])
            for name in _STAT_FIELDS
        }
        self._n_merge = 0

    def start(self) -> None:
        self._loop_t0 = time.perf_counter()
        self.meta["t_start_wall"] = datetime.datetime.now().isoformat(timespec="milliseconds")
        Path(self.stem).parent.mkdir(parents=True, exist_ok=True)
        self._fh_file = open(f"{self.stem}.csv", "w", newline="")
        self._fh = csv.DictWriter(
            self._fh_file, fieldnames=list(self.SCALAR_FIELDS) + list(self.ARRAY_FIELDS),
            extrasaction="ignore",
        )
        self._fh.writeheader()
        # Metadata is static after startup — write it once, O(1).
        self._write_meta(summary=None)
        logger.info("Control log enabled → %s.csv (+ .json meta; summary on close)", self.stem)

    def log(self, **fields) -> None:
        if self._loop_t0 is None:
            return
        fields["t_s"] = time.perf_counter() - self._loop_t0

        counts = self._counts
        counts["n_ticks"] += 1
        popped = bool(fields.get("popped"))
        ik_ok = bool(fields.get("ik_ok"))
        state = fields.get("state")
        counts["n_popped"] += popped
        counts["n_ik_ok"] += ik_ok
        if popped and not ik_ok and state == "INFERENCE":
            counts["n_ik_skip"] += 1
        if not popped and state == "INFERENCE":
            counts["n_underrun"] += 1
        if state == "PAUSED":
            counts["n_paused"] += 1
        for name in _STAT_FIELDS:
            value = fields.get(name)
            if value is not None:
                self._stats[name].update(np.array([[float(value)]], dtype=float))
                self._stat_n[name] += 1

        if self._fh is None:  # log() before start(): keep going with a late start
            self.start()
        self._fh.writerow(self._csv_row(fields, self.SCALAR_FIELDS, self.ARRAY_FIELDS))
        self._fh_file.flush()  # OS cache — survives SIGKILL, no fsync cost

    def log_merge(self, **fields) -> None:
        """Record one temporal-ensemble blend (async only).

        Called by ``ActionBuffer.merge`` for every timestep where an incoming
        chunk overlaps an already-queued absolute target. ``existing_abs`` is the
        target already in the queue (from a prior chunk), ``incoming_abs`` +
        ``ref_ee`` come from the new chunk, ``weight`` is the blend weight, and
        ``aggregated_abs`` is the resulting target stored back. Inspecting these
        confirms the ensemble averages ABSOLUTE targets (same base frame), not
        relative ΔT's averaged directly.
        """
        if self._loop_t0 is None:
            return
        if self._merge_writer is None:
            self._merge_file = open(f"{self.stem}_merge.csv", "w", newline="")
            self._merge_writer = csv.DictWriter(
                self._merge_file, fieldnames=list(self.MERGE_FIELDS), extrasaction="ignore"
            )
            self._merge_writer.writeheader()
        self._n_merge += 1
        self._merge_writer.writerow(self._csv_row(fields, self.MERGE_FIELDS, ()))
        self._merge_file.flush()

    @staticmethod
    def _csv_row(row: dict, scalar_fields, array_fields) -> dict:
        out = {}
        for k in list(scalar_fields) + list(array_fields):
            v = row.get(k)
            if v is None:
                out[k] = ""
            elif isinstance(v, (np.ndarray, list, tuple)):
                out[k] = ";".join(f"{x:.6g}" for x in np.asarray(v).ravel())
            else:
                out[k] = v
        return out

    @staticmethod
    def _stat(qs: RunningQuantileStats, count: int) -> dict | None:
        if count == 0:
            return None
        if count < 2:  # get_statistics needs ≥2 samples; degrade to min/max only
            array = qs._min
            return {"mean": float(array[0]), "std": 0.0, "min": float(array[0]),
                    "max": float(array[0]), "p50": float(array[0]),
                    "p95": float(array[0]), "p99": float(array[0])}
        s = qs.get_statistics()
        return {
            "mean": float(s["mean"]), "std": float(s["std"]),
            "min": float(s["min"]), "max": float(s["max"]),
            "p50": float(s["q50"]), "p95": float(s["q95"]), "p99": float(s["q99"]),
        }

    def _write_meta(self, summary: dict | None) -> None:
        if summary is not None:
            self.meta["summary"] = summary
        Path(f"{self.stem}.json").write_text(json.dumps(self.meta, indent=2, default=str))

    def flush(self) -> None:
        """Durability flush — O(1): just push the CSV append buffer to the OS."""
        if self._fh_file is not None:
            self._fh_file.flush()

    def close(self) -> None:
        empty = self._counts["n_ticks"] == 0
        self.meta["t_end_wall"] = datetime.datetime.now().isoformat(timespec="milliseconds")
        summary = {
            **self._counts,
            **{name: self._stat(self._stats[name], self._stat_n[name]) for name in _STAT_FIELDS},
        }
        self._write_meta(summary=summary)
        for handle in (self._fh_file, self._merge_file):
            if handle is not None:
                handle.close()
        self._fh = self._merge_writer = None
        if empty:
            logger.warning("Control log empty — nothing written")
            return
        merge_note = f", {self._n_merge} ensemble blends" if self._n_merge else ""
        logger.info(
            "Control log saved: %s.csv (%s ticks, %s/%s IK-ok, %s underruns%s)",
            self.stem, summary["n_ticks"], summary["n_ik_ok"], summary["n_popped"],
            summary["n_underrun"], merge_note,
        )


# Curated top-level meta keys — a UNION of the sync and async arg names, read via
# getattr (None where a given script doesn't define that flag). This keeps the
# JSON meta schema identical across both scripts so runs are directly comparable.
_CURATED_META_KEYS: tuple[str, ...] = (
    "pretrained_path", "policy_type", "policy_device", "device",
    "actions_per_chunk", "n_action_steps",
    "chunk_size_threshold", "aggregate_fn_name",
    "max_ee_step_m", "ee_bounds_min", "ee_bounds_max",
    "num_steps", "fps", "warm_start", "gripper_kp", "gripper_kd",
)


def _jsonable(value):
    """Best-effort coerce to JSON-native (lists stay lists; everything else as-is;
    non-native types like Path are handled later by json.dumps(default=str))."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def make_control_logger(args, *, prefix: str = "ctrl", fps: int | None = None) -> ControlLogger | None:
    """Build a :class:`ControlLogger` from ``--log``, or ``None`` when off.

    ``args.log`` is ``None`` (off), ``""`` (bare ``--log`` → auto path under
    ``logs/<prefix>_<timestamp>``), or a caller-supplied stem/path. The top-level
    meta uses a union of sync+async arg names (None where absent) so both scripts
    emit identical meta keys; the full parsed namespace is also dumped under
    ``"args"`` so a run is fully reproducible.
    """
    log = getattr(args, "log", None)
    if log is None:
        return None
    stem = log
    if not stem:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"logs/{prefix}_{stamp}"
    # "foo.csv"/".npz"/".json" → stem "foo"; the writer re-adds the suffixes.
    if stem.endswith((".csv", ".npz", ".json")):
        stem = stem.rsplit(".", 1)[0]
    meta = {k: _jsonable(getattr(args, k, None)) for k in _CURATED_META_KEYS}
    meta["args"] = dict(vars(args))
    return ControlLogger(stem, fps=fps or getattr(args, "fps", 30), meta=meta)

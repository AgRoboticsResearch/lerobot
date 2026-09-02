"""Shared per-tick control logger for the UMI relative-EE Piper deploy scripts.

Both the sync deploy (``deploy_umi_relative_ee_piper.py``) and the async client
(``async_umi_relative_ee_piper_client.py``) import :class:`ControlLogger` /
:func:`make_control_logger` from here so their ``--log`` output shares ONE
schema (same CSV columns, same NPZ arrays, same JSON meta keys) and runs from
either script can be diffed directly.
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class ControlLogger:
    """Per-tick control log for offline stutter / timing analysis (``--log``).

    Off by default; zero overhead when disabled. Each loop tick records the
    timeline, the FULL tick period (work + pacing sleep — so the reported Hz is
    honest rather than the work-only time), loop state, queue depth, the popped
    absolute-EE action (the "trajectory sent to the robot"), the IK joint command
    ("real joint commands"), read-back joints, IK skip events, and inference
    latency snapshots.

    On close() it writes three siblings:
      <stem>.csv   one row per tick (scalars + ';'-joined vectors) — pandas/Excel
      <stem>.npz   clean numeric arrays — numpy / matplotlib
      <stem>.json  run metadata + summary stats (mean dt, #IK skips, #underruns)

    Per-tick UMI relative-EE chunk-provenance fields (added so the temporal
    ensemble can be audited offline — see log_merge for the per-timestep blend):
      chunk_id        which action chunk this tick's target came from
      chunk_ref_ee    7D EE pose that anchored/conditioned that chunk (T_anchor)
      action_abs      7D absolute EE target from that chunk (T_anchor ∘ ΔT), pre-ensemble
      action_agg      7D aggregated absolute target actually executed (post weighted_average)
      action_rel      raw relative action ΔT (model output; SYNC: in-process, ASYNC: server-reported)
      action_ee_clamped 7D post-clamp absolute EE target actually fed to the IK
                      pipeline (position bounds-clipped by EEBoundsAndSafety,
                      which also warns on the console when clipping engages;
                      equals action_ee when no clamp is active; None on
                      rejected ticks)

    And, for async runs only, a fourth sibling records each ensemble blend:
      <stem>_merge.csv  one row per (chunk × overlapping timestep): the existing
                        absolute target, the incoming chunk's absolute target +
                        anchor, the weight, and the resulting aggregated target —
                        so you can confirm the ensemble averages ABSOLUTE targets
                        (same base frame), not relative ΔT's averaged directly.

    A fifth sibling (async, log_chunk) records every incoming chunk including
    its unexecuted tail:
      <stem>_chunks.npz  per chunk: first/last timestep, n actions, replace flag,
                         wire_ms, and the full [max_n, 7] absolute predictions.
    """

    SCALAR_FIELDS: tuple[str, ...] = (
        "step", "tick", "t_s", "tick_dt_ms", "work_ms", "state", "queue",
        "popped", "ik_ok", "skip_reason", "action_timestep", "chunk_id",
        "ee_delta_m", "joint_delta_max_rad", "gripper", "e2e_ms", "wire_ms", "server_ms",
    )
    ARRAY_FIELDS: tuple[str, ...] = (
        "action_ee", "action_ee_clamped", "current_ee", "ik_joints_rad", "current_joints_rad",
        "chunk_ref_ee", "action_abs", "action_agg", "action_rel",
    )
    # Per-blend columns for the async temporal-ensemble audit log (<stem>_merge.csv).
    MERGE_FIELDS: tuple[str, ...] = (
        "step", "tick", "timestep", "chunk_id", "weight",
        "existing_abs", "incoming_abs", "aggregated_abs", "ref_ee",
    )

    def __init__(self, stem: str, *, fps: int, meta: dict, flush_every: int = 150) -> None:
        self.stem = stem
        self.fps = fps
        self.meta = dict(meta)
        self.meta["fps_target"] = fps
        self._rows: list[dict] = []
        self._merge_rows: list[dict] = []
        self._chunk_rows: list[dict] = []
        self._loop_t0: float | None = None
        # Rewrite the files every N ticks so a SIGKILL / OOM / timeout still leaves
        # recent data on disk (within ~flush_every ticks), not just on graceful close.
        self._flush_every = flush_every

    def start(self) -> None:
        self._loop_t0 = time.perf_counter()
        self.meta["t_start_wall"] = datetime.datetime.now().isoformat(timespec="milliseconds")
        logger.info("Control log enabled → %s.{csv,npz,json}", self.stem)

    def log(self, **fields) -> None:
        if self._loop_t0 is None:
            return
        fields["t_s"] = time.perf_counter() - self._loop_t0
        self._rows.append(fields)
        if self._flush_every and len(self._rows) % self._flush_every == 0:
            self._write()

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
        self._merge_rows.append(fields)

    def log_chunk(self, **fields) -> None:
        """Record one incoming action chunk, including the unexecuted tail.

        Called by the async receiver for every accepted response with the FULL
        absolute 7D chunk (``abs_actions`` [n,7] — the whole predicted
        trajectory, most of which is later replaced before execution), its
        timestep span, whether it replaced the queue (RTC) and the wire time.
        Written to ``<stem>_chunks.npz`` so per-tick detail plots can overlay
        each prediction against the commanded/achieved timeline.
        """
        if self._loop_t0 is None:
            return
        self._chunk_rows.append(fields)

    @staticmethod
    def _vec_width(rows: list[dict], name: str) -> int:
        for r in rows:
            v = r.get(name)
            if v is not None:
                return int(np.asarray(v).ravel().shape[0])
        return 0

    def _stack_vec(self, name: str) -> np.ndarray:
        width = self._vec_width(self._rows, name)
        if width == 0:
            return np.empty((0, 0))
        return np.stack([
            np.asarray(r.get(name), dtype=float).ravel()
            if r.get(name) is not None else np.full(width, np.nan)
            for r in self._rows
        ])

    @staticmethod
    def _stat(x: np.ndarray) -> dict | None:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return None
        return {
            "mean": float(np.mean(x)), "std": float(np.std(x)),
            "min": float(np.min(x)), "max": float(np.max(x)),
            "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)),
        }

    def _write(self) -> bool:
        """Rewrite all three sibling files from the buffered rows.

        Called every ``flush_every`` ticks (crash durability) and finally from
        :meth:`close`. Returns False if there is nothing to write yet.
        """
        rows = self._rows
        if not rows:
            return False
        Path(self.stem).parent.mkdir(parents=True, exist_ok=True)

        # --- CSV: scalars as-is, vectors as ';'-joined floats ---
        fields = list(self.SCALAR_FIELDS) + list(self.ARRAY_FIELDS)
        with open(f"{self.stem}.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                out = {}
                for k in fields:
                    v = row.get(k)
                    if v is None:
                        out[k] = ""
                    elif isinstance(v, (np.ndarray, list, tuple)):
                        out[k] = ";".join(f"{x:.6g}" for x in np.asarray(v).ravel())
                    else:
                        out[k] = v
                writer.writerow(out)

        # --- NPZ: clean numeric arrays (NaN where a tick had no action) ---
        scalars: dict[str, np.ndarray] = {}
        for name in ("step", "tick", "t_s", "tick_dt_ms", "work_ms", "queue",
                     "action_timestep", "chunk_id", "ee_delta_m", "joint_delta_max_rad",
                     "gripper", "e2e_ms", "wire_ms", "server_ms"):
            scalars[name] = np.array(
                [float(r.get(name)) if r.get(name) is not None else np.nan for r in rows],
                dtype=float,
            )
        for name in ("popped", "ik_ok"):
            scalars[name] = np.array([1.0 if r.get(name) else 0.0 for r in rows], dtype=float)
        arrays = {name: self._stack_vec(name) for name in self.ARRAY_FIELDS}
        np.savez(f"{self.stem}.npz", **scalars, **arrays)

        # --- JSON: metadata + summary ---
        n_popped = int(scalars["popped"].sum())
        n_ik_ok = int(scalars["ik_ok"].sum())
        self.meta["t_end_wall"] = datetime.datetime.now().isoformat(timespec="milliseconds")
        self.meta["summary"] = {
            "n_ticks": len(rows),
            "n_popped": n_popped,
            "n_ik_ok": n_ik_ok,
            # Real IK failures only: popped + failed + in INFERENCE. Exclude PAUSED
            # ticks (sync pops-but-discards while paused, which is NOT an IK failure),
            # so this stays comparable across the sync and async scripts.
            "n_ik_skip": sum(1 for r in rows
                             if r.get("popped") and not r.get("ik_ok")
                             and r.get("state") == "INFERENCE"),
            "n_underrun": sum(1 for r in rows
                              if not r.get("popped") and r.get("state") == "INFERENCE"),
            "n_paused": sum(1 for r in rows if r.get("state") == "PAUSED"),
            "tick_dt_ms": self._stat(scalars["tick_dt_ms"]),
            "work_ms": self._stat(scalars["work_ms"]),
            "e2e_ms": self._stat(scalars["e2e_ms"]),
            "ee_delta_m": self._stat(scalars["ee_delta_m"]),
            "joint_delta_max_rad": self._stat(scalars["joint_delta_max_rad"]),
        }
        Path(f"{self.stem}.json").write_text(json.dumps(self.meta, indent=2, default=str))

        # --- <stem>_merge.csv: async temporal-ensemble blend audit (if any) ---
        # Snapshot first: _merge_rows is appended by the async receiver thread while
        # _write runs on the main thread (SYNC never populates it, so this is async-only).
        merge_rows = list(self._merge_rows)
        if merge_rows:
            with open(f"{self.stem}_merge.csv", "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(self.MERGE_FIELDS), extrasaction="ignore")
                writer.writeheader()
                for row in merge_rows:
                    out = {}
                    for k in self.MERGE_FIELDS:
                        v = row.get(k)
                        if v is None:
                            out[k] = ""
                        elif isinstance(v, (np.ndarray, list, tuple)):
                            out[k] = ";".join(f"{x:.6g}" for x in np.asarray(v).ravel())
                        else:
                            out[k] = v
                    writer.writerow(out)

        # --- <stem>_chunks.npz: every incoming chunk incl. its unexecuted tail ---
        # Snapshot first: _chunk_rows is appended by the async receiver thread.
        chunk_rows = list(self._chunk_rows)
        if chunk_rows:
            max_n = max(int(r.get("n") or 0) for r in chunk_rows)
            abs_stack = np.full((len(chunk_rows), max_n, 7), np.nan)
            for i, r in enumerate(chunk_rows):
                a = r.get("abs_actions")
                if a is not None:
                    a = np.asarray(a, dtype=float).reshape(-1, 7)
                    abs_stack[i, : len(a)] = a
            np.savez(
                f"{self.stem}_chunks.npz",
                first_ts=np.array([float(r.get("first_ts")) for r in chunk_rows]),
                last_ts=np.array([float(r.get("last_ts")) for r in chunk_rows]),
                n=np.array([float(r.get("n")) for r in chunk_rows]),
                replace=np.array([1.0 if r.get("replace") else 0.0 for r in chunk_rows]),
                wire_ms=np.array([float(r.get("wire_ms") or np.nan) for r in chunk_rows]),
                abs=abs_stack,
            )
        return True

    def flush(self) -> None:
        """Silent periodic durability flush (rewrite all three files)."""
        self._write()

    def close(self) -> None:
        if not self._write():
            logger.warning("Control log empty — nothing written")
            return
        s = self.meta.get("summary", {})
        merge_note = f", {len(self._merge_rows)} ensemble blends" if self._merge_rows else ""
        chunk_note = f", {len(self._chunk_rows)} chunks captured" if self._chunk_rows else ""
        logger.info(
            "Control log saved: %s.{csv,npz,json} (%s ticks, %s/%s IK-ok, %s underruns%s)",
            self.stem, s.get("n_ticks"), s.get("n_ik_ok"), s.get("n_popped"),
            s.get("n_underrun"), merge_note + chunk_note,
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

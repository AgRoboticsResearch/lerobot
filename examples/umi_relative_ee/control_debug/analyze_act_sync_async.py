"""Analyze the same ACT checkpoint under sync and async execution.

This intentionally differs from the older plotting scripts in one important way:
trajectory differences are computed only across consecutive executed loop ticks.
It therefore never treats a pause or underrun as one giant control step.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOGS = REPO / "logs"
OUT = REPO / "outputs" / "research_report" / "low_level_control_debug" / "act_sync_async"

RUNS = {
    "sync": "sync_20260807_150336",
    "async_latest": "async_20260807_150005",
    "async_weighted": "async_20260807_150157",
}


def _load(stem: str) -> tuple[dict[str, np.ndarray], dict]:
    arrays = dict(np.load(LOGS / f"{stem}.npz").items())
    metadata = json.loads((LOGS / f"{stem}.json").read_text())
    return arrays, metadata


def _run_stats(stem: str) -> dict:
    data, metadata = _load(stem)
    valid = np.flatnonzero(data["ik_ok"].astype(bool))
    edge_is_contiguous = np.diff(valid) == 1
    triple_is_contiguous = edge_is_contiguous[:-1] & edge_is_contiguous[1:]
    executed_time_s = float(np.diff(data["t_s"][valid])[edge_is_contiguous].sum())
    executed_segments = int((np.diff(valid) > 1).sum() + 1)

    ee = data["action_ee"][valid, :3].astype(float)
    ee_steps = np.diff(ee, axis=0)
    continuous_steps = ee_steps[edge_is_contiguous]
    step_mm = np.linalg.norm(continuous_steps, axis=1) * 1e3

    reversal_all = np.sum(ee_steps[:-1] * ee_steps[1:], axis=1) < 0
    reversal = reversal_all[triple_is_contiguous]

    chunk = data["chunk_id"][valid]
    chunk_change_edges = chunk[1:] != chunk[:-1]
    continuous_chunk_changes = chunk_change_edges[edge_is_contiguous]
    current_arrival = chunk[2:] != chunk[1:-1]
    previous_arrival = chunk[1:-1] != chunk[:-2]
    reversal_count = int(reversal.sum())

    commanded = data["ik_joints_rad"][valid].astype(float)
    measured = data["current_joints_rad"][valid].astype(float)
    tracking_deg = np.max(np.abs(commanded - measured), axis=1)
    measured_motion_deg = np.max(np.abs(np.diff(measured, axis=0)[edge_is_contiguous]), axis=1)

    start_of_chunk = np.r_[True, chunk[1:] != chunk[:-1]]
    tick_ms = data["tick_dt_ms"][valid]
    summary = metadata["summary"]
    e2e = summary.get("e2e_ms") or {}
    e2e_per_tick = data["e2e_ms"].astype(float)
    e2e_is_finite = np.isfinite(e2e_per_tick)
    e2e_is_new = e2e_is_finite & np.r_[
        True,
        ~np.isclose(e2e_per_tick[1:], e2e_per_tick[:-1], equal_nan=True),
    ]
    e2e_events = e2e_per_tick[e2e_is_new]

    switch_steps = step_mm[continuous_chunk_changes]
    within_steps = step_mm[~continuous_chunk_changes]
    n_continuous_changes = int(continuous_chunk_changes.sum())

    return {
        "stem": stem,
        "checkpoint": metadata["pretrained_path"],
        "mode": "sync" if stem.startswith("sync_") else "async",
        "aggregate": metadata.get("aggregate_fn_name"),
        "chunk_actions": metadata.get("n_action_steps") or metadata.get("actions_per_chunk"),
        "chunk_size_threshold": metadata.get("chunk_size_threshold"),
        "valid_ticks": int(valid.size),
        "executed_time_s": executed_time_s,
        "executed_segments": executed_segments,
        "continuous_edges": int(edge_is_contiguous.sum()),
        "excluded_gap_edges": int((~edge_is_contiguous).sum()),
        "continuous_triples": int(triple_is_contiguous.sum()),
        "chunk_changes_on_continuous_edges": n_continuous_changes,
        "ticks_per_chunk_change": float(edge_is_contiguous.sum() / n_continuous_changes),
        "ee_step_mm": {
            "mean": float(step_mm.mean()),
            "p50": float(np.median(step_mm)),
            "p95": float(np.quantile(step_mm, 0.95)),
            "chunk_change_mean": float(switch_steps.mean()),
            "within_chunk_mean": float(within_steps.mean()),
        },
        "direction_reversal_fraction": float(reversal.mean()),
        "reversal_fraction_at_current_arrival": float(
            reversal_all[triple_is_contiguous & current_arrival].mean()
        ),
        "reversal_fraction_away_from_current_arrival": float(
            reversal_all[triple_is_contiguous & ~current_arrival].mean()
        ),
        "fraction_of_reversals_at_current_or_previous_arrival": float(
            np.sum(reversal_all & triple_is_contiguous & (current_arrival | previous_arrival))
            / reversal_count
        ),
        "joint_tracking_gap_deg": {
            "mean": float(tracking_deg.mean()),
            "p95": float(np.quantile(tracking_deg, 0.95)),
        },
        "measured_joint_motion_deg": {
            "mean": float(measured_motion_deg.mean()),
            "p95": float(np.quantile(measured_motion_deg, 0.95)),
        },
        "tick_ms": {
            "all_p50": float(summary["tick_dt_ms"]["p50"]),
            "chunk_start_p50": float(np.median(tick_ms[start_of_chunk])),
            "within_chunk_p50": float(np.median(tick_ms[~start_of_chunk])),
        },
        "e2e_row_weighted_p50_ms": e2e.get("p50"),
        "e2e_response_events": {
            "count": int(e2e_events.size),
            "p50_ms": float(np.median(e2e_events)) if e2e_events.size else None,
            "p95_ms": float(np.quantile(e2e_events, 0.95)) if e2e_events.size else None,
            "max_ms": float(e2e_events.max()) if e2e_events.size else None,
        },
        "underruns": int(summary["n_underrun"]),
        "ik_skips": int(summary["n_ik_skip"]),
    }


def _merge_stats(stem: str) -> dict:
    with (LOGS / f"{stem}_merge.csv").open() as handle:
        rows = list(csv.DictReader(handle))

    def vectors(field: str) -> np.ndarray:
        return np.asarray([[float(value) for value in row[field].split(";")] for row in rows])

    existing = vectors("existing_abs")
    incoming = vectors("incoming_abs")
    aggregated = vectors("aggregated_abs")
    references = vectors("ref_ee")
    weights = np.asarray([float(row["weight"]) for row in rows])
    chunk_ids = np.asarray([int(row["chunk_id"]) for row in rows])
    unique_chunks = np.unique(chunk_ids)
    per_chunk_reference = np.asarray(
        [references[np.flatnonzero(chunk_ids == chunk_id)[0]] for chunk_id in unique_chunks]
    )

    disagreement_mm = np.linalg.norm(incoming[:, :3] - existing[:, :3], axis=1) * 1e3
    correction_mm = np.linalg.norm(aggregated[:, :3] - existing[:, :3], axis=1) * 1e3
    return {
        "rows": len(rows),
        "chunks": int(unique_chunks.size),
        "weight_on_existing_mean": float(weights.mean()),
        "incoming_existing_translation_mm": {
            "mean": float(disagreement_mm.mean()),
            "p95": float(np.quantile(disagreement_mm, 0.95)),
            "max": float(disagreement_mm.max()),
        },
        "aggregated_existing_translation_mm": {
            "mean": float(correction_mm.mean()),
            "p95": float(np.quantile(correction_mm, 0.95)),
            "max": float(correction_mm.max()),
        },
        "anchor_std_xyz_mm": (np.std(per_chunk_reference[:, :3], axis=0) * 1e3).tolist(),
        "successive_anchor_translation_mm_mean": float(
            np.linalg.norm(np.diff(per_chunk_reference[:, :3], axis=0), axis=1).mean() * 1e3
        ),
    }


def main() -> None:
    results = {name: _run_stats(stem) for name, stem in RUNS.items()}
    results["async_latest"]["merge_audit"] = _merge_stats(RUNS["async_latest"])
    results["async_weighted"]["merge_audit"] = _merge_stats(RUNS["async_weighted"])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "computed_stats.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the kiwi adjacent-frame MSE manifest from canonical result provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_ROOTS = (
    "/mnt/data/zfei/lerobot-act-flow-ablation/eval/jerk_reeval/eval_unified_h10",
    "/mnt/data/zfei/lerobot-act-flow-ablation/eval/salvage_eval/eval_unified_h10",
    "/mnt/data/zfei/lerobot-act-flow-ablation/eval/q3_2frame_h10",
    "/mnt/data/zfei/lerobot-act-flow-ablation/eval/q4_noproprio_h10",
)
ARCHIVE_ROOT = Path("/mnt/data/zfei/lerobot-act-flow-ablation/archive/report_ckpts")


def resolve_checkpoint(raw: str) -> Path:
    checkpoint = Path(raw)
    candidates = [checkpoint]
    marker = "/report_ckpts/"
    if marker in raw:
        candidates.append(ARCHIVE_ROOT / raw.split(marker, maxsplit=1)[1])
    legacy_eval_root = "/mnt/data/zfei/jerk_reeval/"
    if raw.startswith(legacy_eval_root):
        candidates.append(
            Path("/mnt/data/zfei/lerobot-act-flow-ablation/eval/jerk_reeval")
            / raw.removeprefix(legacy_eval_root)
        )
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve checkpoint from {raw}; tried {candidates}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    parser.add_argument(
        "--output",
        default="/mnt/data/zfei/lerobot-act-flow-ablation/eval/cross_frame_mse/manifest.tsv",
    )
    args = parser.parse_args()

    rows: dict[str, tuple[Path, str, Path]] = {}
    for root_string in args.roots:
        root = Path(root_string)
        for report_path in sorted(root.glob("*/seed*/*_open_loop_metrics.json")):
            run = report_path.parents[1].name
            report = json.loads(report_path.read_text())
            checkpoint = resolve_checkpoint(report["checkpoint"])
            config = json.loads((checkpoint / "config.json").read_text())
            policy_type = str(config["type"])
            if run in rows and rows[run][0] != checkpoint:
                raise ValueError(f"Run-name collision for {run}: {rows[run][0]} vs {checkpoint}")
            rows[run] = (checkpoint, policy_type, report_path)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["run\tcheckpoint\tpolicy_type\tsource_report"]
    lines.extend(
        f"{run}\t{checkpoint}\t{policy_type}\t{source}"
        for run, (checkpoint, policy_type, source) in sorted(rows.items())
    )
    output.write_text("\n".join(lines) + "\n")
    heavy = sum(policy_type in {"pi0", "pi05", "smolvla"} for _, policy_type, _ in rows.values())
    print(f"wrote {len(rows)} checkpoints ({len(rows) - heavy} light, {heavy} VLM) to {output}")


if __name__ == "__main__":
    main()

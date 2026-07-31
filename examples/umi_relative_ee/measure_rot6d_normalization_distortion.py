"""Phase 0: quantify the rot6d normalization distortion on real data (no GPU).

Computes the exact per-dimension statistics that the UMI normalizer would fit, then
reports the effective per-dimension "loss weight" under each NormalizationMode and
the imbalance among the rot6d dimensions. This predicts how much per-dim scaling
distorts rotation learning before spending any GPU on the A/B.

Reuses compute_umi_relative_ee_stats so the numbers match the training pipeline.
"""
import argparse

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.umi_relative_ee_stats import compute_umi_relative_ee_stats

ACTION_NAMES = ["dx", "dy", "dz", "rot6d_0", "rot6d_1", "rot6d_2",
                "rot6d_3", "rot6d_4", "rot6d_5", "gripper"]
# rot6d diagonal ("1" near identity) components within the 10D action.
ROT6D_DIAG_IDX = [3, 7]  # rot6d_0 and rot6d_4


def weight_table(stats):
    q01, q99 = stats["q01"], stats["q99"]
    mn, mx = stats["min"], stats["max"]
    std = stats["std"]
    rng_q = q99 - q01
    rng_mm = mx - mn
    return {
        "range_q(q99-q01)": rng_q,
        "w_quantile=1/range": 1.0 / np.where(rng_q == 0, 1e-12, rng_q),
        "range_mm(max-min)": rng_mm,
        "w_minmax=1/range": 1.0 / np.where(rng_mm == 0, 1e-12, rng_mm),
        "w_meanstd=1/std": 1.0 / np.where(std == 0, 1e-12, std),
    }


def fmt_row(name, vals_by_col, col_order):
    cells = [f"{name:>9}"]
    for c in col_order:
        v = vals_by_col[c]
        cells.append(" ".join(f"{x:7.3f}" for x in np.atleast_1d(v)))
    return " | ".join(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="sroi/sroiv2_strawberry_picking_lab_1000onesb")
    ap.add_argument("--root", default="/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb")
    ap.add_argument("--chunk_size", type=int, default=30)
    args = ap.parse_args()

    print(f"Loading {args.repo_id} from {args.root} ...")
    ds = LeRobotDataset(args.repo_id, root=args.root)
    n_frames = len(ds.hf_dataset)
    n_eps = len(np.unique(np.asarray(ds.hf_dataset["episode_index"])))
    print(f"  {n_frames} frames, {n_eps} episodes. chunk_size={args.chunk_size}\n")

    stats = compute_umi_relative_ee_stats(ds.hf_dataset, args.chunk_size, identity_rot6d=False)
    a = stats["action"]
    cols = ["range_q(q99-q01)", "w_quantile=1/range", "range_mm(max-min)",
            "w_minmax=1/range", "w_meanstd=1/std"]
    t = weight_table(a)

    print("Per-dimension ACTION statistics (relative rot6d):")
    print("  " + " | ".join([f"{c:>20}" for c in ["dim"]] + [f"{c:>20}" for c in cols]))
    for i, name in enumerate(ACTION_NAMES):
        row = {c: t[c][i] for c in cols}
        print("  " + " | ".join([f"{name:>20}"] + [f"{row[c]:>20.4f}" for c in cols]))

    rot6d = slice(3, 9)
    pos = slice(0, 3)
    print("\nImbalance summary (QUANTILES weight 1/(q99-q01)):")
    for label, sl in [("pos [0:3]", pos), ("rot6d [3:9]", rot6d), ("  diag (0,4)", ROT6D_DIAG_IDX),
                      ("gripper [9]", 9)]:
        w = t["w_quantile=1/range"][sl] if not isinstance(sl, int) else t["w_quantile=1/range"][sl]
        w = np.atleast_1d(w)
        print(f"  {label:<14} min={w.min():.2f}  max={w.max():.2f}  ratio max/min={w.max()/max(w.min(),1e-12):.1f}x")

    rot6d_w = t["w_quantile=1/range"][rot6d]
    pos_w = t["w_quantile=1/range"][pos]
    print(f"\n  rot6d mean weight / pos mean weight = {rot6d_w.mean()/pos_w.mean():.2f}x")
    print(f"  rot6d diag mean / rot6d offdiag mean = "
          f"{t['w_quantile=1/range'][ROT6D_DIAG_IDX].mean() / np.delete(rot6d_w, [0,3]).mean():.2f}x")

    print("\nInterpretation:")
    print("  - ratio >> 1x on rot6d means per-dim scaling over-weights near-constant")
    print("    rotation dims -> supports H1 (normalization distorts rotation learning).")
    print("  - ratio ~1x means balanced -> mechanism weak, expect H0.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Upgrade the two converted SROI datasets from v3.0 chunked layout to a proper
v2.1 per-episode layout that the pinned openpi LeRobot (commit 0cf864870) reads.

Why: openpi's pinned lerobot is CODEBASE_VERSION "v2.1", which expects
  data_path   = data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet   (one file/episode)
  video_path  = videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4
plus meta/{tasks,episodes,episodes_stats}.jsonl.  The source/converted data is v3.0:
a single chunked data parquet and 13 chunked video files, with parquet metadata.

This script, run once:
  1. ffmpeg-splits the 13 chunked mp4s into 1459 per-episode mp4s (SHARED, since the
     camera frames are identical for rotvec & rot6d). Timestamps reset per episode in
     the data, and from_ts/to_ts are global-within-chunked-file, so a `-ss from_ts
     -frames:v length` cut yields a 0-based per-episode clip matching the data ts.
  2. For each variant dir: splits the converted chunked parquet into per-episode
     parquets; writes tasks.jsonl / episodes.jsonl / episodes_stats.jsonl; patches
     info.json (codebase_version v2.1, data_path/video_path, total_chunks, splits);
     repoints videos/ -> the shared per-episode videos; removes v3.0 meta artifacts.

Idempotent: skips any per-episode file that already exists.
"""
import json, glob, shutil, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/mnt/data1/sroi/lerobot")
SRC = ROOT / "sroiv2_strawberry_picking_lab_1459_occlusion"
VARIANTS = ["sroiv2_strawberry_1459_rotvec", "sroiv2_strawberry_1459_rot6d"]
SHARED_VIDEOS = ROOT / "_sroi_v21_videos"
CAM = "observation.images.camera"
TASK = "pick the strawberry"


def load_episodes_df():
    epf = glob.glob(str(SRC / "meta" / "episodes" / "**" / "*.parquet"), recursive=True)[0]
    import pandas as pd
    return (
        pd.read_parquet(epf)
        .sort_values("episode_index")
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# 1. video reshard (shared)
# --------------------------------------------------------------------------- #
def _ffmpeg_one(job):
    ep, length, from_ts, src, out = job
    out = Path(out)
    if out.exists() and out.stat().st_size > 0:
        return ep, "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{from_ts:.6f}", "-i", str(src),
        "-frames:v", str(int(length)), "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-vsync", "0",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return ep, ("ok" if r.returncode == 0 else f"FAIL[{r.stderr[:160].strip()}]")


def reshard_videos(eps_df, workers=8):
    out_root = SHARED_VIDEOS / CAM
    jobs = []
    for _, r in eps_df.iterrows():
        ep = int(r["episode_index"])
        length = int(r["length"])
        from_ts = float(r[f"videos/{CAM}/from_timestamp"])
        vchunk = int(r[f"videos/{CAM}/chunk_index"])
        vfile = int(r[f"videos/{CAM}/file_index"])
        src = SRC / "videos" / CAM / f"chunk-{vchunk:03d}" / f"file-{vfile:03d}.mp4"
        out = out_root / f"chunk-{ep // 1000:03d}" / f"episode_{ep:06d}.mp4"
        jobs.append((ep, length, from_ts, src, out))
    n_ok = n_skip = n_fail = 0
    fails = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_ffmpeg_one, j) for j in jobs]
        for f in as_completed(futs):
            ep, st = f.result()
            if st == "ok":
                n_ok += 1
            elif st == "skip":
                n_skip += 1
            else:
                n_fail += 1
                fails.append((ep, st))
    print(f"[videos] ok={n_ok} skip={n_skip} fail={n_fail}")
    for ep, st in fails[:10]:
        print(f"   FAIL ep{ep}: {st}")
    return n_fail == 0


# --------------------------------------------------------------------------- #
# 2. per-variant: split data, write meta, link videos
# --------------------------------------------------------------------------- #
def _stats_block(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def build_variant(variant, eps_df):
    d = ROOT / variant
    print(f"\n=== {variant} ===")
    chunked = d / "data" / "chunk-000" / "file-000.parquet"
    tbl = pq.read_table(chunked)
    ep_idx = np.asarray(tbl.column("episode_index").to_pylist())
    uniq = sorted(int(e) for e in np.unique(ep_idx))
    assert len(uniq) == len(eps_df), (len(uniq), len(eps_df))

    act = np.stack(tbl.column("action").to_pylist()).astype(np.float64)
    state = np.stack(tbl.column("observation.state").to_pylist()).astype(np.float64)

    # --- split data parquet per episode ---
    old_data = d / "data"
    tmp_data = d / "_data_v21"
    tmp_data.mkdir(exist_ok=True)
    n_data = 0
    for ep in uniq:
        mask = pa.array((ep_idx == ep).tolist())
        sub = tbl.filter(mask)
        outp = tmp_data / f"chunk-{ep // 1000:03d}" / f"episode_{ep:06d}.parquet"
        outp.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(sub, outp)
        n_data += 1
    shutil.rmtree(old_data)
    (tmp_data).rename(old_data)
    print(f"[data] wrote {n_data} per-episode parquets")

    # --- tasks.jsonl ---
    import jsonlines
    with jsonlines.open(d / "meta" / "tasks.jsonl", "w") as w:
        w.write({"task_index": 0, "task": TASK})

    # --- episodes.jsonl + episodes_stats.jsonl ---
    with jsonlines.open(d / "meta" / "episodes.jsonl", "w") as we, \
         jsonlines.open(d / "meta" / "episodes_stats.jsonl", "w") as ws:
        for ep in uniq:
            length = int(eps_df.loc[eps_df.episode_index == ep, "length"].iloc[0])
            we.write({"episode_index": ep, "tasks": [TASK], "length": length})
            m = ep_idx == ep
            ws.write({
                "episode_index": ep,
                "stats": {
                    "action": _stats_block(act[m]),
                    "observation.state": _stats_block(state[m]),
                },
            })
    print(f"[meta] wrote tasks.jsonl, episodes.jsonl, episodes_stats.jsonl ({len(uniq)} eps)")

    # --- remove v3.0 meta artifacts ---
    for p in [d / "meta" / "tasks.parquet", d / "meta" / "episodes", d / "meta" / "stats.json"]:
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()

    # --- repoint videos/ -> shared per-episode videos ---
    vlink = d / "videos"
    if vlink.is_symlink() or vlink.exists():
        if vlink.is_symlink() or vlink.is_file():
            vlink.unlink()
        else:
            shutil.rmtree(vlink)
    vlink.symlink_to(SHARED_VIDEOS)
    print(f"[videos] -> {SHARED_VIDEOS}")

    # --- patch info.json ---
    ip = d / "meta" / "info.json"
    info = json.load(open(ip))
    info["codebase_version"] = "v2.1"
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4"
    info["total_chunks"] = (int(info["total_episodes"]) + int(info["chunks_size"]) - 1) // int(info["chunks_size"])
    info["splits"] = {"train": f"0:{info['total_episodes']}"}
    json.dump(info, open(ip, "w"), indent=2)
    print(f"[info] patched: v2.1, total_chunks={info['total_chunks']}")


if __name__ == "__main__":
    eps_df = load_episodes_df()
    print(f"loaded {len(eps_df)} episodes from source")
    ok = reshard_videos(eps_df)
    if not ok:
        print("VIDEO RESHARD HAD FAILURES -- aborting before touching data")
        raise SystemExit(1)
    for v in VARIANTS:
        build_variant(v, eps_df)
    print("\nDONE: both datasets upgraded to v2.1 per-episode layout")

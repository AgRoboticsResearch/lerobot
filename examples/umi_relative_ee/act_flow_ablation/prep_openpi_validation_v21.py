#!/usr/bin/env python3
"""Prepare the strawberry validation set as a v2.1 per-episode LeRobot dataset
(rotvec 7D form) so the pinned openpi lerobot can load it for open-loop eval.

Source (v3.0): sroiv2_strawberry_picking_lab_validation -- 100 eps, 9274 frames,
action 7D rotvec, NO observation.state column, video under observation.images.camera.

Output: sroiv2_strawberry_validation_rotvec (v2.1):
  - per-episode parquet (action 7D + observation.state=action 7D + index cols)
  - per-episode mp4 (ffmpeg-split from the single chunked validation video)
  - meta/{tasks,episodes,episodes_stats}.jsonl
  - patched info.json (codebase v2.1, per-episode data_path/video_path, +observation.state)
Single-episode state is later converted rotvec->rot6d on the fly by the rot6d eval.
"""
import json, glob, shutil, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/mnt/data1/sroi/lerobot")
SRC = ROOT / "sroiv2_strawberry_picking_lab_validation"
DST = ROOT / "sroiv2_strawberry_validation_rotvec"
CAM = "observation.images.camera"
TASK = "pick the strawberry"
DIM = 7


def _ffmpeg_one(job):
    ep, length, from_ts, src, out = job
    out = Path(out)
    if out.exists() and out.stat().st_size > 0:
        return ep, "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{from_ts:.6f}", "-i", str(src),
           "-frames:v", str(int(length)), "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-pix_fmt", "yuv420p", "-vsync", "0", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return ep, ("ok" if r.returncode == 0 else f"FAIL[{r.stderr[:120]}]")


def main():
    import pandas as pd
    if (DST / "meta" / "tasks.jsonl").exists():
        print(f"[skip] {DST.name} already complete"); return
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    (DST / "meta").mkdir(parents=True, exist_ok=True)

    eps_df = pd.read_parquet(glob.glob(str(SRC / "meta" / "episodes" / "**" / "*.parquet"), recursive=True)[0]).sort_values("episode_index").reset_index(drop=True)

    # ---- data: per-episode parquet with action + observation.state ----
    tbl = pq.read_table(glob.glob(str(SRC / "data" / "**" / "*.parquet"), recursive=True)[0])
    ep_idx = np.asarray(tbl.column("episode_index").to_pylist())
    uniq = sorted(int(e) for e in np.unique(ep_idx))
    # add observation.state = action
    act_list = tbl.column("action").to_pylist()
    cols = {c: tbl.column(c) for c in tbl.column_names}
    cols["observation.state"] = tbl.column("action")
    full = pa.table(cols)
    act = np.stack(act_list).astype(np.float64)
    n = 0
    for ep in uniq:
        mask = pa.array((ep_idx == ep).tolist())
        sub = full.filter(mask)
        outp = DST / "data" / f"chunk-{ep // 1000:03d}" / f"episode_{ep:06d}.parquet"
        outp.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(sub, outp)
        n += 1
    print(f"[data] {n} per-episode parquets")

    # ---- videos: ffmpeg split (shared single source file) ----
    vroot = DST / "videos" / CAM
    jobs = []
    for _, r in eps_df.iterrows():
        ep = int(r["episode_index"]); length = int(r["length"]); from_ts = float(r[f"videos/{CAM}/from_timestamp"])
        vc = int(r[f"videos/{CAM}/chunk_index"]); vf = int(r[f"videos/{CAM}/file_index"])
        src = SRC / "videos" / CAM / f"chunk-{vc:03d}" / f"file-{vf:03d}.mp4"
        out = vroot / f"chunk-{ep // 1000:03d}" / f"episode_{ep:06d}.mp4"
        jobs.append((ep, length, from_ts, src, out))
    with ProcessPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(_ffmpeg_one, j) for j in jobs]):
            ep, st = f.result()
            if st not in ("ok", "skip"):
                print(f"   video FAIL ep{ep}: {st}")
    print(f"[videos] {len(jobs)} per-episode mp4s")

    # ---- jsonl meta ----
    import jsonlines
    with jsonlines.open(DST / "meta" / "tasks.jsonl", "w") as w:
        w.write({"task_index": 0, "task": TASK})
    def stats_block(a):
        a = np.asarray(a, dtype=np.float64)
        return {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "count": [int(a.shape[0])]}
    with jsonlines.open(DST / "meta" / "episodes.jsonl", "w") as we, jsonlines.open(DST / "meta" / "episodes_stats.jsonl", "w") as ws:
        for ep in uniq:
            length = int(eps_df.loc[eps_df.episode_index == ep, "length"].iloc[0])
            we.write({"episode_index": ep, "tasks": [TASK], "length": length})
            m = ep_idx == ep
            ws.write({"episode_index": ep, "stats": {"action": stats_block(act[m]), "observation.state": stats_block(act[m])}})
    print("[meta] tasks/episodes/episodes_stats jsonl")

    # ---- info.json (v2.1 + observation.state feature) ----
    info = json.load(open(SRC / "meta" / "info.json"))
    info["codebase_version"] = "v2.1"
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4"
    info["total_chunks"] = (int(info["total_episodes"]) + int(info["chunks_size"]) - 1) // int(info["chunks_size"])
    info["splits"] = {"train": f"0:{info['total_episodes']}"}
    # add observation.state feature mirroring action
    info["features"]["observation.state"] = json.loads(json.dumps(info["features"]["action"]))
    json.dump(info, open(DST / "meta" / "info.json", "w"), indent=2)
    print(f"[info] v2.1, total_chunks={info['total_chunks']}, +observation.state")
    print("DONE:", DST)


if __name__ == "__main__":
    main()

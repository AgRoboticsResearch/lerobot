#!/usr/bin/env python
"""Re-stamp a converted LeRobot dataset from its nominal fps to the true rate.

The h5 sources carry no timestamps; recordings were triggered by a camera/
command callback at ~30 Hz, not the nominal 50 Hz rospy rate. This fixes all
fps-derived metadata consistently, with no re-encoding:

  1. videos: ffmpeg -itsscale old/new stream-copy remux (frame N moves from
     N/old to N/new seconds; bitstream untouched)
  2. data parquet:   timestamp = frame_index / new_fps
  3. episodes parquet: videos/*/to_timestamp = length / new_fps;
     per-episode stats/timestamp/* recomputed
  4. meta/stats.json: global timestamp stats recomputed
  5. meta/info.json: fps; manifest.json/README.md annotated

Everything is staged to *.new files and swapped in only after every step of
the dataset succeeded, so a crash leaves the dataset at the old fps intact.
Re-running is a no-op once info.json already shows the new fps.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

QUANTS = ("q01", "q10", "q50", "q90", "q99")


def agg(values: np.ndarray) -> dict:
    v = values.astype(np.float64)
    return {
        "count": [len(v)],
        "min": [float(v.min())],
        "max": [float(v.max())],
        "mean": [float(v.mean())],
        "std": [float(v.std())],
        **{q: [float(np.quantile(v, float(q[1:]) / 100))] for q in QUANTS},
    }


def restamp(ds_dir: Path, old_fps: int, new_fps: int, note: str):
    info_path = ds_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if info["fps"] == new_fps:
        print(f"[{ds_dir.name}] already {new_fps} fps — skipping")
        return
    if info["fps"] != old_fps:
        sys.exit(f"[{ds_dir.name}] unexpected fps {info['fps']}, expected {old_fps}")
    factor = old_fps / new_fps

    # ---- phase 1: remux all videos to .new --------------------------------
    # Two-step stream copy so frame N moves from N/old to N/new seconds EXACTLY
    # (zero pts rounding; required because pyav decode matches parquet float32
    # timestamps against container pts within 1e-4 s):
    #   A: timescale 12800 -> 384000 (x30, exact; input files all use 1/12800)
    #   B: -itsscale old/new at 1/384000 granularity (k*7680*old/new is an
    #      integer for old/new=5/3) onto the final 1/90000 base
    vids = sorted(ds_dir.glob("videos/**/*.mp4"))
    for i, v in enumerate(vids):
        tmp = Path(str(v) + ".new")
        step_a = Path(str(v) + ".stepA")
        for cmd in (
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "mp4", "-i", str(v),
             "-c", "copy", "-video_track_timescale", "384000", "-f", "mp4", str(step_a)],
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "mp4",
             "-itsscale", repr(factor), "-i", str(step_a),
             "-c", "copy", "-video_track_timescale", "90000", "-f", "mp4", str(tmp)],
        ):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                for u in ds_dir.glob("videos/**/*.mp4.new"):
                    u.unlink()
                step_a.unlink(missing_ok=True)
                sys.exit(f"[{ds_dir.name}] ffmpeg failed on {v}: {r.stderr}")
        step_a.unlink()
        if i % 500 == 0:
            print(f"[{ds_dir.name}] remuxed {i}/{len(vids)}")
    print(f"[{ds_dir.name}] remuxed {len(vids)}/{len(vids)} videos")

    # ---- phase 2: rewrite data parquet timestamps to .new ------------------
    all_ts = []
    for f in sorted(ds_dir.glob("data/**/*.parquet")):
        t = pq.read_table(f)
        ts = (t.column("frame_index").to_numpy() / new_fps).astype(np.float32)
        all_ts.append(ts)
        cols = {n: t.column(n) for n in t.column_names}
        cols["timestamp"] = pa.array(ts, type=t.schema.field("timestamp").type)
        pq.write_table(pa.table(cols), str(f) + ".new")

    # ---- phase 3: episodes parquet to .new ---------------------------------
    # from_timestamp = start offset WITHIN the episode's video file (videos are
    # chunked: N episodes per file, offsets restart per file) = cum_prev_len_in_
    # that file / fps; to_timestamp = episode duration in the video = length/fps
    ep_files = sorted(ds_dir.glob("meta/episodes/**/*.parquet"))
    cum: dict = {}  # (camera, chunk_index, file_index) -> frames so far in that video file
    for f in ep_files:
        t = pq.read_table(f)
        d = sorted(t.to_pylist(), key=lambda r: r["episode_index"])
        for row in d:
            L = row["length"]
            s = agg((np.arange(L) / new_fps).astype(np.float32))
            for col in list(row):
                if col.endswith("/from_timestamp"):
                    prefix, c, fi = col[:-len("/from_timestamp")], \
                        row[col[: -len("from_timestamp")] + "chunk_index"], \
                        row[col[: -len("from_timestamp")] + "file_index"]
                    g = (prefix, c, fi)
                    row[col] = cum.get(g, 0) / new_fps
                    cum[g] = cum.get(g, 0) + L
                elif col.endswith("/to_timestamp"):
                    row[col] = L / new_fps
                elif col.startswith("stats/timestamp/"):
                    # these are list columns in the episodes schema
                    key = col.split("/", 2)[2]
                    val = s[key][0]
                    row[col] = [int(val)] if key == "count" else [val]
        pq.write_table(pa.Table.from_pylist(d, schema=t.schema), str(f) + ".new")

    # ---- phase 4: global stats.json ----------------------------------------
    stats_path = ds_dir / "meta" / "stats.json"
    stats_new = Path(str(stats_path) + ".new")
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        if "timestamp" in stats:
            stats["timestamp"] = agg(np.concatenate(all_ts))
        stats_new.write_text(json.dumps(stats, indent=2))

    # ---- phase 5: info.json to .new ----------------------------------------
    info["fps"] = new_fps
    Path(str(info_path) + ".new").write_text(json.dumps(info, indent=2))

    # ---- swap everything in -------------------------------------------------
    for v in vids:
        Path(str(v) + ".new").replace(v)
    for f in sorted(ds_dir.glob("data/**/*.parquet.new")):
        f.replace(str(f)[:-4])
    for f in ep_files:
        Path(str(f) + ".new").replace(f)
    if stats_new.exists():
        stats_new.replace(stats_path)
    Path(str(info_path) + ".new").replace(info_path)

    # ---- manifest / README --------------------------------------------------
    mf_path = ds_dir / "manifest.json"
    if mf_path.exists():
        mf = json.loads(mf_path.read_text())
        mf["fps"] = new_fps
        mf["fps_note"] = note
        mf_path.write_text(json.dumps(mf, indent=2))
    rd = ds_dir / "README.md"
    if rd.exists():
        txt = rd.read_text().replace(f"fps={old_fps}", f"fps={new_fps}")
        txt += f"\n> fps re-stamped {old_fps} -> {new_fps}: {note}\n"
        rd.write_text(txt)
    print(f"[{ds_dir.name}] DONE: {len(vids)} videos remuxed, fps {old_fps}->{new_fps}")


if __name__ == "__main__":
    restamp(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),
            sys.argv[4] if len(sys.argv) > 4 else "corrected to true capture rate")

#!/usr/bin/env python3
"""Content-address a LeRobot dataset for the reproducibility bundle.

Policy: every small deterministic artifact (meta/, tasks, parquet shards) is
hashed file-by-file with sha256; the video payload (the bulk of the bytes) is
NOT hashed file-by-file by default — instead a manifest of (relative path,
size, mtime_ns) is hashed into a single digest. Pass --full to also sha256
every video (slow on large sets).

Usage: hash_dataset.py <dataset_root> [--full] > <name>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--full", action="store_true", help="hash videos too (slow)")
    args = ap.parse_args()

    files = {}
    videos_manifest = {}
    video_hashes = {}
    for dirpath, _dirnames, filenames in os.walk(args.root):
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            rel = os.path.relpath(p, args.root)
            st = os.stat(p)
            if name.endswith((".mp4", ".webm", ".mkv", ".avi")):
                videos_manifest[rel] = [st.st_size, st.st_mtime_ns]
                if args.full:
                    video_hashes[rel] = sha256_file(p)
            else:
                files[rel] = [st.st_size, sha256_file(p)]

    man_digest = hashlib.sha256(
        json.dumps(videos_manifest, sort_keys=True).encode()
    ).hexdigest()
    info_path = os.path.join(args.root, "meta", "info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else None
    out = {
        "root": args.root,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "full_video_hash": args.full,
        "meta_info": info,
        "num_hashed_files": len(files),
        "num_videos": len(videos_manifest),
        "video_bytes_total": sum(v[0] for v in videos_manifest.values()),
        "videos_manifest_sha256": man_digest,
        "files": dict(sorted(files.items())),
    }
    if args.full:
        out["video_hashes"] = dict(sorted(video_hashes.items()))
    json.dump(out, sys.stdout, indent=1, sort_keys=True)
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pre-convert the strawberry-1459 LeRobot dataset into two rotation-notation
variants for the official-openpi LoRA experiment.

The stored `action` is already start-anchored relative-EE (frame-0 ~= 0):
xyz[3] + rotvec(rotation)[3] + gripper[1]. So NO delta math is needed.

  * rotvec variant: action unchanged (7D). Add observation.state = action (7D).
  * rot6d variant: action = concat(xyz, rot6d(rotvec), gripper) -> 10D, using the
    UMI row-based rot6d convention (first two ROWS of the rotation matrix) to
    match the lerobot port. Add observation.state = action (10D).

openpi requires a `state` key (the source has none), so we derive it per frame
as the (converted) action -- openpi's delta_timestamps then pulls the current
frame as state and the future chunk as actions. Datasets land as siblings under
/mnt/data1/sroi/lerobot/ so HF_LEROBOT_HOME=/mnt/data1/sroi/lerobot resolves them.
Videos are symlinked (shared, read-only); stats.json is dropped so openpi's
compute_norm_stats regenerates it.
"""
import json, shutil, os
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

SRC = "/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1459_occlusion"
DST_ROOT = "/mnt/data1/sroi/lerobot"


def rotvec_to_rot6d(rot):  # (N,3) rotvec -> (N,6) UMI row convention (first 2 rows)
    M = R.from_rotvec(rot).as_matrix()        # (N,3,3)
    return M[:, :2, :].reshape(M.shape[0], 6)  # first two rows -> 6


def convert(variant, out_name):
    dst = os.path.join(DST_ROOT, out_name)
    if os.path.exists(dst):
        print(f"[skip] {out_name} exists"); return
    os.makedirs(dst)
    dim = 10 if variant == "rot6d" else 7
    # videos: symlink (shared, read-only)
    os.symlink(os.path.join(SRC, "videos"), os.path.join(dst, "videos"))
    # meta: copytree (handles episodes/ dir), patch info.json, drop stats.json
    shutil.copytree(os.path.join(SRC, "meta"), os.path.join(dst, "meta"))
    if os.path.exists(os.path.join(dst, "meta", "stats.json")):
        os.remove(os.path.join(dst, "meta", "stats.json"))  # regenerate via openpi compute_norm_stats
    # root helpers
    for f in ("merge_info.json", "MERGE_INFO.md"):
        if os.path.exists(os.path.join(SRC, f)):
            shutil.copy(os.path.join(SRC, f), os.path.join(dst, f))
    # convert parquet(s)
    src_data = os.path.join(SRC, "data")
    for pf in sorted(__import__("glob").glob(os.path.join(src_data, "**", "*.parquet"), recursive=True)):
        rel = os.path.relpath(pf, src_data)
        outpf = os.path.join(dst, "data", rel)
        os.makedirs(os.path.dirname(outpf), exist_ok=True)
        t = pq.read_table(pf)
        act = np.stack(t.column("action").to_pylist()).astype(np.float32)  # (N,7)
        assert act.shape[1] == 7, act.shape
        xyz, rot, grip = act[:, :3], act[:, 3:6], act[:, 6:7]
        newact = np.concatenate([xyz, rotvec_to_rot6d(rot), grip], axis=1) if variant == "rot6d" else act
        newact = newact.astype(np.float32)
        action_arr = pa.array([r.tolist() for r in newact], type=pa.list_(pa.float32(), dim))
        cols = {c: t.column(c) for c in t.column_names if c != "action"}
        cols["action"] = action_arr
        cols["observation.state"] = action_arr  # per-frame state = current (relative) pose
        pq.write_table(pa.table(cols), outpf)
        print(f"[{out_name}] wrote {outpf} rows={t.num_rows} dim={dim}")
    # patch info.json
    ip = os.path.join(dst, "meta", "info.json")
    info = json.load(open(ip))
    feat = info["features"]
    feat["action"]["shape"] = [dim]
    if variant == "rot6d":
        feat["action"]["names"] = ["ee.x", "ee.y", "ee.z"] + [f"r6d.{i}" for i in range(6)] + ["ee.gripper_pos"]
    # add observation.state (mirror action)
    state_feat = json.loads(json.dumps(feat["action"]))
    feat["observation.state"] = state_feat
    json.dump(info, open(ip, "w"), indent=2)
    print(f"[{out_name}] patched info.json: action.shape=[{dim}], added observation.state")


if __name__ == "__main__":
    convert("rotvec", "sroiv2_strawberry_1459_rotvec")
    convert("rot6d", "sroiv2_strawberry_1459_rot6d")
    print("done")

#!/usr/bin/env python
"""Convert ALOHA/ACT-style HDF5 demos into LeRobot v3.0 datasets.

One invocation converts ONE LeRobot dataset from an explicit list of source
session directories. Datasets are organized per recording campaign (never
merged just because camera sets match) so they can be re-combined later from
the provenance manifest.

Source layout (per session dir, episode files found recursively):
    <session>/episode_*.hdf5           (episode_54_val.hdf5 = validation split)
        /action                     (T, A) float32   joint cmds (+ gripper)
        /observations/qpos          (T, Q) float32   measured joints
        [/observations/qvel]        (T, Q) float32   (z1 / bimanual sim data)
        [/observations/effort]      (T, Q) float32   (z1 / bimanual sim data)
        /observations/images/<cam>  (T, 480, 640[, 3]) uint8 (+ *_infra2 mono IR)

Joint vectors are written as float32 (bit-exact). Videos use lerobot's default
encoding (libsvtav1/av1, crf 30, preset 12, g 2, yuv420p). The sources carry
no timestamps; fps (default 50, the nominal rospy.Rate(50) rate) synthesizes
them. Frames measured as stored-BGR are flipped to RGB on read (see
RGB_EXCEPTIONS).

Example — the sim-env-block-pick priority dataset:
    .venv/bin/python convert_scara_h5_to_lerobot.py \
        --name scara_sim_env_block_pick --skip-val \
        --session <...>/2024-05-14_07-31-04-sim-env-block-pick --alias data1 \
        --session <...>/2024-05-16_06-51-09--sim-env-block-pick --alias data2 \
        ...
"""
import argparse
import glob
import json
import re
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_OUT = "/mnt/data1/strawberry_robot/lerobot_datasets"

# The h5 frames were recorded through a recorder that applied cvtColor(BGR2RGB)
# unconditionally (robot_teleoperation/act_demo_real_scara.py) while the camera
# drivers mostly delivered rgb8, so nearly all stored frames are channel-
# swapped (BGR in an RGB-labeled array). Measured per session/camera by
# counting red-dominant vs blue-dominant pixel clusters (fruit is red).
# Entries here were already stored RGB and must NOT be flipped:
RGB_EXCEPTIONS = {
    "2024-04-29_03-37-35_1_sb_3cam": {"wrist"},
    # sim recorders wrote renderer output directly (no cvtColor) -> stored RGB
    "sim_transfer_cube_scripted": {"top", "left_wrist", "right_wrist"},
    "sim_transfer_cube_s1": {"top", "left_wrist", "right_wrist"},
}

VEC_NAMES = {  # dimension -> feature component names
    4: ["j1", "j2", "j3", "j4"],
    5: ["j1", "j2", "j3", "j4", "gripper"],
    6: ["j1", "j2", "j3", "j4", "j5", "j6"],
    14: ["l_j1", "l_j2", "l_j3", "l_j4", "l_j5", "l_j6", "l_gripper",
         "r_j1", "r_j2", "r_j3", "r_j4", "r_j5", "r_j6", "r_gripper"],
}


def episode_sort_key(path: Path):
    """episode_0 < episode_1 < ... < episode_10; ties by name (val last)."""
    m = re.match(r"episode_(\d+)(.*)", path.stem)
    return (int(m.group(1)), m.group(2)) if m else (10**9, path.name)


def find_episodes(session: Path) -> list[Path]:
    return sorted(session.rglob("episode_*.hdf5"), key=episode_sort_key)


def probe_schema(h5_path: Path, include_infra: bool = False) -> dict:
    """Feature schema of one episode file (vector dims, cameras, frame count)."""
    with h5py.File(h5_path, "r") as f:
        obs = f["observations"]
        cams = sorted(
            k for k in obs["images"].keys()
            if include_infra or not k.endswith("_infra2")
        )
        return {
            "qpos": obs["qpos"].shape[1],
            "action": f["action"].shape[1],
            "qvel": obs["qvel"].shape[1] if "qvel" in obs else None,
            "effort": obs["effort"].shape[1] if "effort" in obs else None,
            "cameras": cams,
            # mono IR frames are replicated to 3 channels for yuv420p on write
            "img_shapes": {
                c: tuple([*obs["images"][c].shape[1:], 3][:3])
                if obs["images"][c].ndim == 3 else tuple(obs["images"][c].shape[1:])
                for c in cams
            },
        }


def vec_feature(names: list[str]) -> dict:
    return {"dtype": "float32", "shape": (len(names),), "names": list(names)}


def build_features(schema: dict) -> dict:
    def names(dim, is_action):
        if dim in VEC_NAMES:
            return VEC_NAMES[dim]
        return [f"j{i+1}" for i in range(dim)] if not is_action else \
               [f"a{i+1}" for i in range(dim)]
    features = {
        "observation.state": vec_feature(names(schema["qpos"], False)),
        "action": vec_feature(names(schema["action"], True)),
    }
    for skey, fkey in (("qvel", "observation.qvel"), ("effort", "observation.effort")):
        if schema[skey] is not None:
            features[fkey] = vec_feature(names(schema[skey], False))
    for cam in schema["cameras"]:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": schema["img_shapes"][cam],
            "names": ["height", "width", "channels"],
        }
    return features


def convert_episode(dataset, h5_path: Path, task: str, flip_cams: set[str],
                    avail: set[str], shapes: dict[str, tuple]) -> int:
    """Stream one h5 episode into the dataset; returns frame count.

    Cameras in CAMERA_ORDER but not in `avail` are synthesized as black frames
    (`shapes` holds their (h, w, 3)) so sessions with different camera sets can
    share one output schema.
    """
    with h5py.File(h5_path, "r") as f:
        obs = f["observations"]
        qpos, action = obs["qpos"][:], f["action"][:]
        qvel = obs["qvel"][:] if "qvel" in obs else None
        effort = obs["effort"][:] if "effort" in obs else None
        n = len(qpos)
        images = {c: (obs[f"images/{c}"] if c in avail else None) for c in CAMERA_ORDER}
        for t in range(n):
            frame = {
                "observation.state": qpos[t].astype(np.float32),
                "action": action[t].astype(np.float32),
                "task": task,
            }
            if qvel is not None:
                frame["observation.qvel"] = qvel[t].astype(np.float32)
            if effort is not None:
                frame["observation.effort"] = effort[t].astype(np.float32)
            for cam, ds in images.items():
                if ds is None:  # camera absent in this session -> black fill
                    img = np.zeros(shapes[cam], dtype=np.uint8)
                else:
                    img = ds[t]
                    if img.ndim == 2:  # mono IR -> 3 channels for yuv420p
                        img = np.repeat(img[:, :, None], 3, axis=2)
                    if cam in flip_cams:  # BGR -> RGB
                        img = np.ascontiguousarray(img[..., ::-1])
                frame[f"observation.images.{cam}"] = img
            dataset.add_frame(frame)
    dataset.save_episode()
    return n


def dir_size_gb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e9


def verify(name: str, manifest: dict, out_parent: Path):
    """Fast integrity check: bit-compare vectors vs every source h5 (decode-free),
    then spot-decode one video frame per camera."""
    ds_dir = out_parent / name
    tables = [pq.read_table(f) for f in sorted(glob.glob(str(ds_dir / "data/**/*.parquet"), recursive=True))]
    states = np.concatenate([t.column("observation.state").to_numpy().tolist() for t in tables]).astype(np.float32)
    actions = np.concatenate([t.column("action").to_numpy().tolist() for t in tables]).astype(np.float32)
    assert len(states) == manifest["frames"], f"{name}: {len(states)} != {manifest['frames']} frames"

    offset = 0
    for s in manifest["sessions"]:
        for ep in s["episodes"]:
            with h5py.File(Path(s["path"]) / ep["file"], "r") as f:
                q, a = f["observations/qpos"][:], f["action"][:]
            n = ep["frames"]
            assert np.array_equal(states[offset:offset + n], q), f"{name}/{s['alias']}/{ep['file']}: STATE mismatch"
            assert np.array_equal(actions[offset:offset + n], a), f"{name}/{s['alias']}/{ep['file']}: ACTION mismatch"
            offset += n

    ds = LeRobotDataset(repo_id=name, root=ds_dir)
    mid = len(ds) // 2
    for cam in manifest["cameras"]:
        img = ds[mid][f"observation.images.{cam}"]
        shape = manifest["img_shapes"][cam]
        assert tuple(img.shape) == (shape[2], *shape[:2]), f"{name}/{cam}: {img.shape}"

    # black-filled (synthetic) cameras decode as all-zero; real ones do not
    syn_sessions = [s for s in manifest["sessions"] if s.get("synthetic_cameras") and s["episodes"]]
    if syn_sessions:
        s = syn_sessions[0]
        base = sum(e["frames"] for p in manifest["sessions"][:manifest["sessions"].index(s)]
                   for e in p["episodes"])
        item = ds[base + s["episodes"][0]["frames"] // 2]
        for cam in s["synthetic_cameras"]:
            assert float(item[f"observation.images.{cam}"].float().abs().sum()) == 0.0, \
                f"{name}/{s['alias']}/{cam}: synthetic camera is not black"
    if syn_sessions or any(s.get("synthetic_cameras") for s in manifest["sessions"]):
        assert any(float(ds[mid][f"observation.images.{c}"].float().sum()) > 0
                   for c in manifest["cameras"]), f"{name}: every camera black at mid frame?"

    info = json.loads((ds_dir / "meta" / "info.json").read_text())
    vi = info["features"][f"observation.images.{manifest['cameras'][0]}"]["info"]
    print(f"[{name}] VERIFY OK: {manifest['episodes']} eps / {manifest['frames']} frames, vectors bit-exact; "
          f"codec={vi['video.codec']} crf={vi['video.crf']} g={vi['video.g']} pix={vi['video.pix_fmt']}")


def write_readme(name: str, manifest: dict, out_parent: Path):
    lines = [
        f"# {name}\n",
        f"LeRobot v3.0 dataset converted from ACT/ALOHA HDF5 demos on {manifest['created']}.",
        f"fps={manifest['fps']}; videos av1/crf30/g2/yuv420p (lerobot defaults); "
        f"vectors float32 bit-exact; stored-BGR frames flipped to RGB (see manifest "
        f"`flipped_bgr_to_rgb` per session).\n",
        f"- Episodes: **{manifest['episodes']}** ({manifest['frames']:,} frames, {manifest['output_gb']:.2f} GB)",
        f"- Split: **{manifest['split']}**"
        + (f" ({manifest['skipped_val']} validation episodes excluded — see companion `_val` dataset)" if manifest["split"] == "train" else ""),
        f"- Cameras: {', '.join(manifest['cameras'])}",
        f"- Vector features: state {manifest['dims']['qpos']}D"
        + (f", qvel {manifest['dims']['qvel']}D" if manifest['dims']['qvel'] else "")
        + (f", effort {manifest['dims']['effort']}D" if manifest['dims']['effort'] else "")
        + f", action {manifest['dims']['action']}D\n",
        "## Source sessions\n",
        "| Alias | Source session | Episodes (converted / total) | Frames |",
        "|---|---|---|---|",
    ]
    for s in manifest["sessions"]:
        total = len(s["episodes"]) + len(s["skipped_val_files"])
        frames = sum(e["frames"] for e in s["episodes"])
        lines.append(f"| `{s['alias']}` | `{s['path']}` | {len(s['episodes'])} / {total} | {frames:,} |")
    lines.append("\nPer-episode provenance (file names, frame counts, skipped val files) is in `manifest.json`.\n")
    (out_parent / name / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="dataset name (directory under --out)")
    parser.add_argument("--session", action="append", required=True, help="source session dir (repeatable)")
    parser.add_argument("--alias", action="append", default=None,
                        help="label for the preceding --session, e.g. data1 (repeatable, optional)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="parent dir for datasets")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--skip-val", action="store_true", help="exclude episode_*_val.hdf5 files (train split)")
    parser.add_argument("--only-val", action="store_true",
                        help="convert ONLY episode_*_val.hdf5 files (companion val dataset)")
    parser.add_argument("--include-infra", action="store_true", help="also convert *_infra2 mono IR streams")
    parser.add_argument("--cameras", default="",
                        help="comma-separated camera subset/order for the output schema (default: all found)")
    parser.add_argument("--fill-cameras", default="",
                        help="comma-separated cameras to black-fill in episodes that lack them "
                             "(requires --cameras; enables merging sessions with different camera sets)")
    parser.add_argument("--max-episodes", type=int, default=0, help="cap total episodes (0 = all)")
    parser.add_argument("--erase", action="store_true", help="delete existing dataset dir first")
    parser.add_argument("--no-flip", dest="flip", action="store_false", help="don't fix BGR->RGB")
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    args = parser.parse_args()

    if args.alias and len(args.alias) != len(args.session):
        sys.exit("--alias count must match --session count")

    out_parent = Path(args.out)
    ds_dir = out_parent / args.name
    if ds_dir.exists():
        if not args.erase:
            sys.exit(f"{ds_dir} already exists (use --erase to replace it)")
        print(f"erasing existing {ds_dir}")
        shutil.rmtree(ds_dir)
    out_parent.mkdir(parents=True, exist_ok=True)

    # schema: vector dims must match everywhere; camera availability may vary
    # ACROSS sessions (with --fill-cameras) but not within one
    sessions = []
    for i, s in enumerate(args.session):
        p = Path(s).resolve()
        eps = find_episodes(p)
        if not eps:
            sys.exit(f"no episode_*.hdf5 under {p}")
        sessions.append({"path": str(p), "root": p, "alias": args.alias[i] if args.alias else p.name,
                         "files": eps})
    requested = [c.strip() for c in args.cameras.split(",") if c.strip()] or None
    fill = {c.strip() for c in args.fill_cameras.split(",") if c.strip()}
    if fill and requested is None:
        sys.exit("--fill-cameras requires --cameras")

    dims, session_cams, shapes = None, [], {}
    for s in sessions:
        s_cams = None
        for ep in s["files"]:
            pr = probe_schema(ep, args.include_infra)
            d = {k: pr[k] for k in ("qpos", "action", "qvel", "effort")}
            if dims is None:
                dims = d
            elif d != dims:
                sys.exit(f"schema mismatch across sessions — refusing to merge:\n"
                         f"  {s['alias']}/{ep.name}: {d}\n  vs first episode: {dims}")
            if s_cams is None:
                s_cams = pr["cameras"]
            elif pr["cameras"] != s_cams:
                sys.exit(f"camera set varies inside session {s['alias']}: {ep.name}")
            shapes.update({c: tuple(pr["img_shapes"][c]) for c in pr["cameras"]})
        session_cams.append(set(s_cams))

    if requested:
        for s, avail in zip(sessions, session_cams):
            extra = avail - set(requested)
            if extra:
                sys.exit(f"session {s['alias']} has cameras not in --cameras: {sorted(extra)}")
            missing = set(requested) - avail
            if missing - fill:
                sys.exit(f"session {s['alias']} lacks cameras {sorted(missing - fill)} "
                         f"(add them to --fill-cameras)")
        cameras = requested
    else:
        cameras = sorted(set().union(*session_cams) if session_cams else set())
    assert dims is not None  # sessions are non-empty (checked above)
    for c in cameras:  # every output camera needs a known shape
        shapes.setdefault(c, (480, 640, 3))
    schema = {**dims, "cameras": cameras,
              "img_shapes": {c: list(shapes[c]) for c in cameras}}
    print(f"dataset {args.name}: schema {schema['qpos']}D state / {schema['action']}D action, "
          f"cameras {schema['cameras']}, {len(sessions)} sessions")

    global CAMERA_ORDER
    CAMERA_ORDER = list(schema["cameras"])

    dataset = LeRobotDataset.create(
        repo_id=args.name, fps=args.fps, features=build_features(schema), root=ds_dir,
    )

    manifest = {
        "dataset": args.name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "fps": args.fps,
        "dims": {"qpos": schema["qpos"], "action": schema["action"],
                 "qvel": schema["qvel"], "effort": schema["effort"]},
        "cameras": schema["cameras"], "img_shapes": {c: list(s) for c, s in schema["img_shapes"].items()},
        "split": "val" if args.only_val else ("train" if args.skip_val else "all"),
        "skipped_val": 0,
        "sessions": [], "episodes": 0, "frames": 0,
    }
    done = 0
    t0 = time.time()
    for i, s in enumerate(sessions):
        avail = session_cams[i] & set(schema["cameras"])
        flip = (avail - RGB_EXCEPTIONS.get(Path(s["path"]).name, set())) if args.flip else set()
        s_entry = {"alias": s["alias"], "path": s["path"], "flipped_bgr_to_rgb": sorted(flip),
                   "synthetic_cameras": [c for c in schema["cameras"] if c not in avail],
                   "episodes": [], "skipped_val_files": []}
        for ep in s["files"]:
            is_val = ep.stem.endswith("_val")
            if is_val:
                s_entry["skipped_val_files"].append(str(ep.relative_to(s["root"])))
                if args.skip_val:
                    manifest["skipped_val"] += 1
                    continue
            elif args.only_val:
                continue
            if args.max_episodes and done >= args.max_episodes:
                break
            n = convert_episode(dataset, ep, task=s["alias"], flip_cams=flip,
                                avail=avail, shapes=shapes)
            s_entry["episodes"].append({"file": str(ep.relative_to(s["root"])), "frames": n})
            manifest["episodes"] += 1
            manifest["frames"] += n
            done += 1
            if done % 25 == 0:
                print(f"[{args.name}] {done} episodes, {manifest['frames']} frames, "
                      f"{time.time() - t0:.0f}s elapsed")
        manifest["sessions"].append(s_entry)

    dataset.finalize()
    manifest["duration_s"] = round(time.time() - t0, 1)
    manifest["output_gb"] = round(dir_size_gb(ds_dir), 3)
    (ds_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_readme(args.name, manifest, out_parent)
    print(f"[{args.name}] done: {manifest['episodes']} episodes ({manifest['skipped_val']} val skipped), "
          f"{manifest['frames']} frames, {manifest['output_gb']} GB, {manifest['duration_s']}s")

    if args.verify:
        verify(args.name, manifest, out_parent)


if __name__ == "__main__":
    main()

"""Experiment contracts, geometry, immutable provenance, and image-free motion extraction."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_TRAIN = "/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1459_occlusion"
DEFAULT_VAL = "/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation"
DEFAULT_CHECKPOINT = (
    "/mnt/data1/projects/lerobot-arch-exp/lerobot-arch-exp/train/"
    "act_r18_l1_seed1000_100000steps/checkpoints/100000/pretrained_model"
)
METHODS = ("ik", "residual", "query", "hindsight")
CONDITIONS = ("native", "delay40")


@dataclass(frozen=True)
class Config:
    train_root: str = DEFAULT_TRAIN
    val_root: str = DEFAULT_VAL
    task_checkpoint: str = DEFAULT_CHECKPOINT
    seed: int = 1000
    train_rollouts: int = 10000
    dev_rollouts: int = 1000
    starts_per_query: int = 3
    query_limit: int = 0
    steps: int = 30000
    eval_freq: int = 5000
    batch_size: int = 256
    width: int = 256
    encoder_layers: int = 4
    decoder_layers: int = 2
    heads: int = 8
    feedforward: int = 1024
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    ee_hz: int = 30
    control_hz: int = 50
    physics_hz: int = 500
    horizon: int = 30
    history: int = 11
    command_horizon: int = 5
    lower_xyz: tuple = (-0.5, -0.5, -0.1)
    upper_xyz: tuple = (0.5, 0.5, 0.6)
    max_ee_step: float = 0.05
    max_joint_vel_deg_s: float = 90.0
    joint_margin_deg: float = 5.0
    bootstrap_resamples: int = 10000
    smoke: bool = False

    @classmethod
    def smoke_config(cls, **kwargs):
        return cls(
            train_rollouts=8,
            dev_rollouts=2,
            starts_per_query=1,
            query_limit=2,
            steps=4,
            eval_freq=2,
            batch_size=8,
            width=32,
            encoder_layers=1,
            decoder_layers=1,
            feedforward=64,
            bootstrap_resamples=200,
            smoke=True,
            **kwargs,
        )

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def manifest(path: Path, expected: dict, resume: bool) -> None:
    """Bind a stage to its inputs before writing anything; never mix configurations."""
    if path.exists():
        if not resume:
            raise FileExistsError(f"{path} exists; pass --resume to continue identical inputs")
        if json.loads(path.read_text()) != json.loads(json.dumps(expected)):
            raise ValueError(f"Provenance mismatch: {path}")
    else:
        write_json(path, expected)


def poses_from_actions(actions: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    out = np.broadcast_to(np.eye(4), (len(actions), 4, 4)).copy()
    out[:, :3, 3] = actions[:, :3]
    out[:, :3, :3] = Rotation.from_rotvec(actions[:, 3:6]).as_matrix()
    return out


def relative_poses(poses: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    return np.linalg.inv(anchor) @ poses


def pose_features(poses: np.ndarray) -> np.ndarray:
    # UMI stores the first two ROWS, not the first two columns.
    return np.concatenate((poses[..., :3, 3], poses[..., :2, :3].reshape(*poses.shape[:-2], 6)), -1)


def interpolate(times: np.ndarray, poses: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation, Slerp

    times, query = np.asarray(times, dtype=float), np.asarray(query, dtype=float)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("At least two strictly increasing trajectory timestamps required")
    valid = (query >= times[0] - 1e-8) & (query <= times[-1] + 1e-8)
    clamped = np.clip(query, times[0], times[-1])
    result = np.broadcast_to(np.eye(4), (len(query), 4, 4)).copy()
    result[:, :3, 3] = np.stack([np.interp(clamped, times, poses[:, i, 3]) for i in range(3)], -1)
    result[:, :3, :3] = Slerp(times, Rotation.from_matrix(poses[:, :3, :3]))(clamped).as_matrix()
    return result, valid


def rotation_errors(actual: np.ndarray, desired: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    delta = np.swapaxes(desired[..., :3, :3], -1, -2) @ actual[..., :3, :3]
    return np.rad2deg(Rotation.from_matrix(delta).magnitude())


def read_episodes(root: Path) -> dict[int, dict]:
    import pyarrow.parquet as pq

    info = json.loads((root / "meta/info.json").read_text())
    if info["fps"] != 30 or info["features"]["action"]["shape"] != [7]:
        raise ValueError(f"Expected 30 Hz, absolute 7D UMI actions: {root}")
    pieces = {}
    for file in sorted(root.glob("data/chunk-*/*.parquet")):
        rows = pq.read_table(file, columns=["episode_index", "frame_index", "timestamp", "action"])
        ids = rows["episode_index"].to_numpy()
        frames = rows["frame_index"].to_numpy()
        times = rows["timestamp"].to_numpy()
        actions = np.asarray(rows["action"].to_pylist(), dtype=np.float64)
        for eid in np.unique(ids):
            select = ids == eid
            pieces.setdefault(int(eid), []).append((frames[select], times[select], actions[select]))
    episodes = {}
    for eid, parts in pieces.items():
        frames, times, actions = (np.concatenate([p[i] for p in parts]) for i in range(3))
        order = np.argsort(frames)
        if not np.array_equal(frames[order], np.arange(len(frames))):
            raise ValueError(f"Noncontiguous episode {eid}")
        episodes[eid] = {"times": times[order], "actions": actions[order]}
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Episode count differs from metadata")
    return episodes


def source_fingerprint(root: Path) -> dict:
    paths = [root / "meta/info.json", *sorted(root.glob("data/chunk-*/*.parquet"))]
    return {str(p.relative_to(root)): digest(p) for p in paths}


def extract(root: Path, cfg: Config, resume: bool) -> None:
    if Path(cfg.train_root).resolve() == Path(cfg.val_root).resolve():
        raise ValueError("Training and final validation roots must differ")
    episodes = read_episodes(Path(cfg.train_root))
    shuffled = np.random.default_rng(cfg.seed).permutation(sorted(episodes))
    ndev = max(1, round(len(shuffled) * 0.1))
    splits = {"dev": sorted(map(int, shuffled[:ndev])), "train": sorted(map(int, shuffled[ndev:]))}
    meta = {"config": cfg.to_dict(), "splits": splits, "source": source_fingerprint(Path(cfg.train_root))}
    manifest(root / "motions/manifest.json", meta, resume)
    for eid, ep in episodes.items():
        path = root / f"motions/episodes/{eid:06d}.npz"
        if not path.exists():
            save_npz(path, **ep)
    write_json(root / "motions/complete.json", {"episodes": len(episodes), "manifest": json_hash(meta)})


def motion_from_episode(ep: dict, rng, augment=True):
    from scipy.spatial.transform import Rotation

    actions, times = ep["actions"], ep["times"]
    eligible = np.flatnonzero(times <= times[-1] - 29 / 30 - 1e-6)
    if not len(eligible):
        raise ValueError("Episode too short for a full motion chunk")
    frame = int(rng.choice(eligible))
    poses = poses_from_actions(actions[frame : frame + 30])
    rel = relative_poses(poses, poses[0])
    scale_p, scale_r, speed = rng.uniform(0.8, 1.2, 3) if augment else (1.0, 1.0, 1.0)
    rel[:, :3, 3] *= scale_p
    rel[:, :3, :3] = Rotation.from_rotvec(
        Rotation.from_matrix(rel[:, :3, :3]).as_rotvec() * scale_r
    ).as_matrix()
    return {
        "relative": rel,
        "times": (times[frame : frame + 30] - times[frame]) / speed,
        "gripper": actions[frame : frame + 30, 6],
        "frame": frame,
        "augmentation": [float(scale_p), float(scale_r), float(speed)],
    }


def learning_window(rollout: dict, index: int, cfg: Config, hindsight: bool) -> dict:
    """State is measured before u[index]; targets are issued commands, never delayed deliveries."""
    h, k = cfg.history, cfg.command_horizon
    ncommands = len(rollout["issued"])
    if index < h - 1 or index >= ncommands:
        raise IndexError(index)
    future_times = rollout["time"][index] + np.arange(cfg.horizon) / cfg.ee_hz
    poses, valid = interpolate(
        rollout["time"] if hindsight else rollout["query_time"],
        rollout["actual"] if hindsight else rollout["query"],
        future_times,
    )
    end = min(index + k, ncommands)
    target = np.zeros((k, 6), dtype=np.float32)
    target[: end - index] = rollout["issued"][index:end]
    return {
        "trajectory": pose_features(poses).astype(np.float32),
        "trajectory_valid": valid,
        "state": np.concatenate(
            (rollout["q"][index - h + 1 : index + 1], rollout["qd"][index - h + 1 : index + 1]), -1
        ).astype(np.float32),
        "past_commands": rollout["issued"][index - h + 1 : index].astype(np.float32),
        "target": target,
        "target_valid": np.arange(k) < end - index,
    }

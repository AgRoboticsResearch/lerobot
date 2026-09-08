"""Matched direct-controller pilots, selected solely on embodiment development rollouts."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from examples.umi_relative_ee.task_independent_embodiment.core import digest, manifest, write_json
from examples.umi_relative_ee.task_independent_embodiment.learning import save_checkpoint

ROOTS = {
    "piper": Path("/mnt/data1/projects/lerobot-embodiment-exp"),
    "so101": Path("/mnt/data1/projects/lerobot-embodiment-exp-so101"),
}
NAMESPACES = {"piper": "task_independent_embodiment", "so101": "so101_task_independent_embodiment"}


@dataclass(frozen=True)
class PilotConfig:
    robot: str = "piper"
    representation: str = "command_delta"
    labels: str = "hindsight"
    seed: int = 1000
    steps: int = 2000
    eval_freq: int = 1000
    batch: int = 256
    width: int = 256
    layers: int = 4
    development_rollouts: int = 64
    condition: str = "native"


def modules(robot):
    prefix = "examples.umi_relative_ee." + NAMESPACES[robot]
    return tuple(importlib.import_module(prefix + "." + part) for part in ("core", "learning", "simulation"))


def rotations(features):
    """Decode row-rot6d with orthogonalization, matching the repository convention."""
    a = features[..., 3:6]
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = features[..., 6:9]
    b = b - (a * b).sum(-1, keepdims=True) * a
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    return np.stack((a, b, np.cross(a, b)), axis=-2)


def transform(trajectory, state, past_commands, current_pose, representation):
    """Pure transform shared by materialization, teacher forcing, and live execution."""
    n = past_commands.shape[-1]
    q = state[..., -1, :n].copy()
    last_command = past_commands[..., -1, :].copy()
    trajectory = trajectory.copy()
    state, past_commands = state.copy(), past_commands.copy()
    if representation == "absolute":
        base = np.zeros_like(q)
    elif representation in ("q_delta", "command_delta"):
        base = q if representation == "q_delta" else last_command
        rotation = current_pose[..., :3, :3]
        translation = trajectory[..., :3] - current_pose[..., None, :3, 3]
        rel_p = np.einsum("...ji,...hj->...hi", rotation, translation)
        rel_r = np.einsum("...ji,...hjk->...hik", rotation, rotations(trajectory))
        trajectory = np.concatenate((rel_p, rel_r[..., :2, :].reshape(*rel_p.shape[:-1], 6)), -1)
        state[..., :n] -= q[..., None, :]
        past_commands -= q[..., None, :]
    else:
        raise ValueError(representation)
    # The t=0 hindsight target is tautologically the measured current pose. Never
    # expose that token: runtime tracking error would violate this training invariant.
    trajectory[..., 0, :] = 0
    return {
        "trajectory": trajectory.astype(np.float32),
        "state": state.astype(np.float32),
        "past_commands": past_commands.astype(np.float32),
        "q_reference": q.astype(np.float32),
        "base": base.astype(np.float32),
    }


class DirectControllerModel(nn.Module):
    def __init__(self, cfg: PilotConfig, joints):
        super().__init__()
        self.cfg, self.joints = cfg, joints
        self.projections = nn.ModuleDict(
            {
                k: nn.Linear(v, cfg.width)
                for k, v in {
                    "trajectory": 9,
                    "state": joints * 2,
                    "past_commands": joints,
                    "q_reference": joints,
                }.items()
            }
        )
        self.position = nn.Parameter(torch.randn(1, 52, cfg.width) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, 5, cfg.width) * 0.02)
        self.transformer = nn.Transformer(
            d_model=cfg.width,
            nhead=8,
            num_encoder_layers=cfg.layers,
            num_decoder_layers=2,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.head = nn.Linear(cfg.width, joints)
        for key, width in {
            "trajectory": 9,
            "state": 2 * joints,
            "past_commands": joints,
            "q_reference": joints,
            "target": joints,
        }.items():
            self.register_buffer(key + "_mean", torch.zeros(width))
            self.register_buffer(key + "_std", torch.ones(width))

    def set_statistics(self, stats):
        for key, values in stats.items():
            for stat in ("mean", "std"):
                getattr(self, key + "_" + stat).copy_(torch.tensor(values[stat]))
        if self.cfg.representation != "absolute":
            # Exact physical zero offset at initialization, despite nonzero label means.
            nn.init.zeros_(self.head.weight)
            with torch.no_grad():
                self.head.bias.copy_(-self.target_mean / self.target_std)

    def normalize(self, key, value):
        return (value - getattr(self, key + "_mean")) / getattr(self, key + "_std")

    def forward(self, batch):
        parts = []
        for key in self.projections:
            value = batch[key].unsqueeze(1) if key == "q_reference" else batch[key]
            parts.append(self.projections[key](self.normalize(key, value)))
        tokens = torch.cat(parts, 1) + self.position
        valid = batch["trajectory_valid"].clone()
        valid[:, 0] = False
        mask = torch.cat(
            (
                ~valid,
                torch.zeros(
                    len(tokens),
                    22,
                    dtype=torch.bool,
                    device=tokens.device,
                ),
            ),
            1,
        )
        features = self.transformer(
            tokens,
            self.queries.expand(len(tokens), -1, -1),
            src_key_padding_mask=mask,
            memory_key_padding_mask=mask,
        )
        return self.head(features)

    def predict(self, batch):
        return self(batch) * self.target_std + self.target_mean + batch["base"].unsqueeze(1)


class Controller:
    method = "hindsight"  # adapter must never call its residual IK path

    def __init__(self, model, representation, device):
        self.model, self.representation, self.device = model, representation, device

    @torch.inference_mode()
    def __call__(self, trajectory, valid, state, past, first_pose, ik):
        valid = valid.copy()
        valid[0] = False
        if not valid.any():
            return past[-1].copy()  # no supported positive-time target remains
        current_pose = ik.fk(state[-1, : past.shape[-1]])  # FK only; no inverse-kinematics solve
        values = transform(trajectory, state, past, current_pose, self.representation)
        values["trajectory_valid"] = valid
        batch = {
            key: torch.as_tensor(value, device=self.device).unsqueeze(0) for key, value in values.items()
        }
        return self.model.predict(batch)[0, 0].cpu().numpy()


class HoldController:
    method = "hindsight"

    def __init__(self, kind="command"):
        self.kind = kind

    def __call__(self, trajectory, valid, state, past, first_pose, ik):
        return past[-1].copy() if self.kind == "command" else state[-1, : past.shape[-1]].copy()


def prepare(root, cfg):
    legacy_root = ROOTS[cfg.robot]
    source = legacy_root / f"prepared/{cfg.condition}/C0/{cfg.labels}"
    collection = legacy_root / f"collections/{cfg.condition}/D0/train"
    if not (source / "complete.json").exists():
        raise ValueError(f"Original preparation not complete: {source}")
    directory = root / f"prepared/{cfg.robot}/{cfg.condition}/{cfg.labels}/{cfg.representation}"
    evidence = {
        "source_manifest": digest(source / "manifest.json"),
        "source_completion": digest(source / "complete.json"),
        "collection_manifest": digest(collection / "manifest.json"),
        "representation": cfg.representation,
        "code": digest(Path(__file__)),
    }
    manifest(directory / "manifest.json", evidence, resume=True)
    if (directory / "complete.json").exists():
        return directory
    original = {p.stem: np.load(p, mmap_mode="r") for p in source.glob("*.npy")}
    n, joints = len(original["target"]), original["target"].shape[-1]
    shapes = {k: v.shape for k, v in original.items()}
    shapes.update(q_reference=(n, joints), base=(n, joints))
    arrays = {
        key: np.lib.format.open_memmap(
            directory / f"{key}.npy",
            mode="w+",
            shape=shape,
            dtype=bool if key.endswith("valid") else np.float32,
        )
        for key, shape in shapes.items()
    }
    offset = 0
    for file in sorted(collection.glob("rollouts/*.npz")):
        with np.load(file) as roll:
            current = roll["actual"][10:-1]
        end = offset + len(current)
        sl = slice(offset, end)
        transformed = transform(
            original["trajectory"][sl],
            original["state"][sl],
            original["past_commands"][sl],
            current,
            cfg.representation,
        )
        for key, value in transformed.items():
            arrays[key][sl] = value
        arrays["trajectory_valid"][sl] = original["trajectory_valid"][sl]
        arrays["trajectory_valid"][sl, 0] = False
        arrays["target_valid"][sl] = original["target_valid"][sl]
        arrays["target"][sl] = original["target"][sl] - transformed["base"][:, None, :]
        offset = end
    if offset != n:
        raise ValueError("Source window order/count differs from original preparation")
    eligible = np.flatnonzero(np.asarray(arrays["trajectory_valid"]).any(1))
    np.save(directory / "eligible.npy", eligible)
    stats = {}
    for key in ("trajectory", "state", "past_commands", "q_reference", "target"):
        size = arrays[key].shape[-1]
        count, total, squares = 0, np.zeros(size), np.zeros(size)
        for begin in range(0, len(eligible), 4096):
            ids = eligible[begin : begin + 4096]
            part = np.asarray(arrays[key][ids], dtype=float)
            if key in ("trajectory", "target"):
                part = part[arrays[key + "_valid"][ids]]
            part = part.reshape(-1, size)
            count += len(part)
            total += part.sum(0)
            squares += (part**2).sum(0)
        mean = total / count
        std = np.sqrt(np.maximum(squares / count - mean**2, 0)).clip(1e-3)
        stats[key] = {"mean": mean.tolist(), "std": std.tolist()}
    for array in arrays.values():
        array.flush()
    write_json(directory / "statistics.json", stats)
    write_json(directory / "complete.json", {"windows": n, "eligible_windows": len(eligible)})
    return directory


def development_files(cfg):
    files = sorted((ROOTS[cfg.robot] / f"collections/{cfg.condition}/D0/dev/rollouts").glob("*.npz"))
    rng = np.random.default_rng(90210)
    indices = np.sort(rng.choice(len(files), min(len(files), cfg.development_rollouts), replace=False))
    return [files[i] for i in indices]


def evaluate(controller, files, sim, legacy_cfg, output=None):
    rows = []
    for file in files:
        with np.load(file) as source:
            motion = {
                "relative": source["motion_relative"],
                "times": source["motion_times"],
                "gripper": source["gripper_query"],
            }
            result, feasible = sim.rollout(motion, source["start"], controller)
        metrics = modules(controller.robot)[2].rollout_metrics(result, legacy_cfg)
        rows.append({"rollout": file.name, **metrics, **feasible})
    means = {key: float(np.mean([row[key] for row in rows])) for key in metrics}
    if output:
        write_json(output, {"means": means, "trials": rows})
    return means


def train(root, cfg, device):
    core, legacy_learning, simulation = modules(cfg.robot)
    legacy_cfg = core.Config.read(ROOTS[cfg.robot] / "config.json")
    prepared = prepare(root, cfg)
    name = f"{cfg.representation}_{cfg.labels}_seed{cfg.seed}_{cfg.steps}steps"
    directory = root / f"train/{cfg.robot}/{cfg.condition}/{name}"
    files = development_files(cfg)
    evidence = {
        "config": asdict(cfg),
        "prepared": digest(prepared / "manifest.json"),
        "development": {p.name: digest(p) for p in files},
        "code": digest(Path(__file__)),
        "legacy_preflight": digest(ROOTS[cfg.robot] / "preflight.json"),
    }
    manifest(directory / "manifest.json", evidence, resume=True)
    if (directory / "complete.json").exists():
        return json.loads((directory / "complete.json").read_text())
    arrays = {p.stem: np.load(p, mmap_mode="r") for p in prepared.glob("*.npy") if p.stem != "eligible"}
    eligible = np.load(prepared / "eligible.npy")
    joints = arrays["target"].shape[-1]
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = DirectControllerModel(cfg, joints).to(device)
    model.set_statistics(json.loads((prepared / "statistics.json").read_text()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sim = simulation.Simulation(legacy_cfg, cfg.condition)
    # Save matched hold and original-model baselines on the SAME development subset.
    baselines_path = (
        root / f"development_baselines/{cfg.robot}/{cfg.condition}/{cfg.development_rollouts}.json"
    )
    if not baselines_path.exists():
        baseline_results = {}
        for label, controller in (
            ("hold_command", HoldController("command")),
            ("hold_q", HoldController("q")),
            (
                "v1_hindsight",
                legacy_learning.LearnedController(
                    ROOTS[cfg.robot] / f"train/{cfg.condition}/C0/hindsight_seed1000/best.pt", device
                ),
            ),
            (
                "v1_residual",
                legacy_learning.LearnedController(
                    ROOTS[cfg.robot] / f"train/{cfg.condition}/C0/residual_seed1000/best.pt", device
                ),
            ),
        ):
            controller.robot = cfg.robot
            baseline_results[label] = evaluate(controller, files, sim, legacy_cfg)
        write_json(baselines_path, {"files": [p.name for p in files], "means": baseline_results})
    # A first run may construct baseline models while later runs reuse their
    # reports. That must not alter the training dropout RNG stream.
    torch.manual_seed(cfg.seed + 1)
    first, best = 0, float("inf")
    last = directory / "last.pt"
    if last.exists():
        payload = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        first, best = payload["step"], payload["best"]
        rng.bit_generator.state = payload["numpy_rng"]
        torch.set_rng_state(payload["torch_rng"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
    for step in range(first + 1, cfg.steps + 1):
        model.train()
        ids = rng.choice(eligible, cfg.batch)
        batch = {k: torch.from_numpy(np.asarray(v[ids])).to(device) for k, v in arrays.items()}
        prediction = model(batch)
        mask = batch["target_valid"].unsqueeze(-1)
        loss = ((prediction - model.normalize("target", batch["target"])).abs() * mask).sum() / (
            mask.sum() * joints
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"v2 {cfg.robot}/{name} step={step} loss={loss.item():.6f}", flush=True)
        if step % cfg.eval_freq == 0 or step == cfg.steps:
            model.eval()
            controller = Controller(model, cfg.representation, device)
            controller.robot = cfg.robot
            with torch.random.fork_rng(
                devices=list(range(torch.cuda.device_count())) if device.startswith("cuda") else []
            ):
                metrics = evaluate(
                    controller, files, sim, legacy_cfg, directory / f"development/{step:07d}.json"
                )
            score = metrics["position_rmse_m"]
            improved = score < best
            best = min(best, score)
            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": asdict(cfg),
                "joints": joints,
                "step": step,
                "best": best,
                "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
                "metrics": metrics,
                "provenance": evidence,
            }
            save_checkpoint(last, payload)
            if improved:
                save_checkpoint(directory / "best.pt", payload)
            print(f"v2 development {cfg.robot}/{name}: {json.dumps(metrics)}", flush=True)
    summary = {"best_position_rmse_m": best, "steps": cfg.steps, "checkpoint": str(directory / "best.pt")}
    write_json(directory / "complete.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/data1/projects/lerobot-embodiment-v2"))
    parser.add_argument("--robot", choices=ROOTS, default="piper")
    parser.add_argument(
        "--representation", choices=("absolute", "q_delta", "command_delta"), default="command_delta"
    )
    parser.add_argument("--labels", choices=("hindsight", "query"), default="hindsight")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--development-rollouts", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg = PilotConfig(
        robot=args.robot,
        representation=args.representation,
        labels=args.labels,
        steps=args.steps,
        eval_freq=args.eval_freq,
        development_rollouts=args.development_rollouts,
    )
    train(args.root, cfg, args.device)


if __name__ == "__main__":
    main()

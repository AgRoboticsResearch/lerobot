"""Matched image-free transformers and resumable, bounded-memory training."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .core import Config, digest, json_hash, learning_window, manifest, write_json


class EmbodimentTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.pose_input = nn.Linear(9, cfg.width)
        self.state_input = nn.Linear(2 * cfg.arm_joints, cfg.width)
        self.command_input = nn.Linear(cfg.arm_joints, cfg.width)
        tokens = cfg.horizon + cfg.history * 2 - 1
        self.position = nn.Parameter(torch.randn(1, tokens, cfg.width) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, cfg.command_horizon, cfg.width) * 0.02)
        self.transformer = nn.Transformer(
            d_model=cfg.width,
            nhead=cfg.heads,
            num_encoder_layers=cfg.encoder_layers,
            num_decoder_layers=cfg.decoder_layers,
            dim_feedforward=cfg.feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.output = nn.Linear(cfg.width, cfg.arm_joints)
        for key, size in (
            ("trajectory", 9),
            ("state", 2 * cfg.arm_joints),
            ("past_commands", cfg.arm_joints),
            ("target", cfg.arm_joints),
        ):
            self.register_buffer(key + "_mean", torch.zeros(size))
            self.register_buffer(key + "_std", torch.ones(size))

    def normalize(self, key, x):
        return (x - getattr(self, key + "_mean")) / getattr(self, key + "_std")

    def set_statistics(self, statistics):
        for key, values in statistics.items():
            for stat in ("mean", "std"):
                getattr(self, key + "_" + stat).copy_(torch.tensor(values[stat], dtype=torch.float32))

    def forward(self, batch):
        tokens = (
            torch.cat(
                (
                    self.pose_input(self.normalize("trajectory", batch["trajectory"])),
                    self.state_input(self.normalize("state", batch["state"])),
                    self.command_input(self.normalize("past_commands", batch["past_commands"])),
                ),
                dim=1,
            )
            + self.position
        )
        padding = torch.cat(
            (
                ~batch["trajectory_valid"],
                torch.zeros(
                    len(tokens),
                    self.cfg.history * 2 - 1,
                    dtype=torch.bool,
                    device=tokens.device,
                ),
            ),
            dim=1,
        )
        features = self.transformer(
            tokens,
            self.queries.expand(len(tokens), -1, -1),
            src_key_padding_mask=padding,
            memory_key_padding_mask=padding,
        )
        return self.output(features)

    def predict(self, batch):
        return self(batch) * self.target_std + self.target_mean


class LearnedController:
    def __init__(self, checkpoint: Path, device: str):
        # Only locally produced experiment checkpoints are accepted here.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = Config(**payload["config"])
        self.method = payload["method"]
        self.device = torch.device(device)
        self.model = EmbodimentTransformer(self.cfg).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, trajectory, valid, state, past, first_pose, ik):
        values = {"trajectory": trajectory, "trajectory_valid": valid, "state": state, "past_commands": past}
        batch = {
            k: torch.as_tensor(
                np.asarray(v), device=self.device, dtype=torch.bool if k.endswith("valid") else torch.float32
            ).unsqueeze(0)
            for k, v in values.items()
        }
        result = self.model.predict(batch)[0, 0].cpu().numpy()
        if self.method == "residual":
            result = result + ik.command(first_pose, state[-1, : self.cfg.arm_joints])[0]
        return result


def collection_files(directory: Path) -> list[Path]:
    if not (directory / "complete.json").exists():
        raise ValueError(f"Collection incomplete: {directory}")
    return sorted(directory.glob("rollouts/*.npz"))


def prepare(directory: Path, collections: list[Path], method: str, cfg: Config) -> Path:
    """Materialize mmap windows once per method/generation; never normalize from development data."""
    files = [f for source in collections for f in collection_files(source)]
    evidence = {str(f): digest(f) for f in files}
    expected = {"inputs": evidence, "method": method, "config": cfg.to_dict()}
    manifest(directory / "manifest.json", expected, resume=True)
    if (directory / "complete.json").exists():
        return directory
    counts = []
    for path in files:
        with np.load(path) as rollout:
            counts.append(max(0, len(rollout["issued"]) - cfg.history + 1))
    total = sum(counts)
    if not total:
        raise ValueError("No valid training windows")
    shapes = {
        "trajectory": (cfg.horizon, 9),
        "trajectory_valid": (cfg.horizon,),
        "state": (cfg.history, 2 * cfg.arm_joints),
        "past_commands": (cfg.history - 1, cfg.arm_joints),
        "target": (cfg.command_horizon, cfg.arm_joints),
        "target_valid": (cfg.command_horizon,),
    }
    arrays = {
        key: np.lib.format.open_memmap(
            directory / f"{key}.npy",
            mode="w+",
            shape=(total, *shape),
            dtype=bool if key.endswith("valid") else np.float32,
        )
        for key, shape in shapes.items()
    }
    offset = 0
    for path in files:
        with np.load(path) as loaded:
            rollout = dict(loaded)
        for index in range(cfg.history - 1, len(rollout["issued"])):
            window = learning_window(rollout, index, cfg, hindsight=method != "query")
            if method == "residual":
                n = min(cfg.command_horizon, len(rollout["issued"]) - index)
                window["target"][:n] -= rollout["nominal_actual"][index : index + n]
                window["target_valid"][:n] &= rollout["nominal_valid"][index : index + n]
            for key, array in arrays.items():
                array[offset] = window[key]
            offset += 1
    for array in arrays.values():
        array.flush()
    statistics = {}
    for key in ("trajectory", "state", "past_commands", "target"):
        size = shapes[key][-1]
        count, summed, squared = 0, np.zeros(size), np.zeros(size)
        for begin in range(0, total, 4096):
            part = np.asarray(arrays[key][begin : begin + 4096], dtype=np.float64)
            if key in ("trajectory", "target"):
                part = part[arrays[key + "_valid"][begin : begin + 4096]]
            part = part.reshape(-1, size)
            count += len(part)
            summed += part.sum(0)
            squared += (part**2).sum(0)
        if not count:
            raise ValueError(f"No valid {key} samples")
        mean = summed / count
        std = np.sqrt(np.maximum(squared / count - mean**2, 0)).clip(1e-3)
        statistics[key] = {"mean": mean.tolist(), "std": std.tolist()}
    write_json(directory / "statistics.json", statistics)
    write_json(directory / "complete.json", {"windows": total, "manifest": json_hash(expected)})
    return directory


def save_checkpoint(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(
    directory: Path,
    prepared: Path,
    method: str,
    cfg: Config,
    seed: int,
    device: str,
    evaluate,
    resume: bool,
    evaluation_identity: dict,
):
    expected = {
        "config": cfg.to_dict(),
        "method": method,
        "seed": seed,
        "data": digest(prepared / "manifest.json"),
        "development": evaluation_identity,
    }
    manifest(directory / "manifest.json", expected, resume)
    if (directory / "complete.json").exists():
        return
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = EmbodimentTransformer(cfg).to(device)
    model.set_statistics(json.loads((prepared / "statistics.json").read_text()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    arrays = {p.stem: np.load(p, mmap_mode="r") for p in prepared.glob("*.npy")}
    n = len(arrays["target"])
    first_step, best_score = 0, (float("inf"), float("inf"))
    last = directory / "last.pt"
    if resume and last.exists():
        payload = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        first_step, best_score = payload["step"], tuple(payload["best_score"])
        rng.bit_generator.state = payload["numpy_rng"]
        torch.set_rng_state(payload["torch_rng"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all([s.cpu() for s in payload["cuda_rng"]])
    for step in range(first_step + 1, cfg.steps + 1):
        model.train()
        indices = rng.integers(0, n, cfg.batch_size)
        batch = {
            key: torch.from_numpy(np.asarray(value[indices])).to(device) for key, value in arrays.items()
        }
        prediction = model(batch)
        loss_values = (prediction - model.normalize("target", batch["target"])).abs()
        mask = batch["target_valid"].unsqueeze(-1)
        loss = (loss_values * mask).sum() / (mask.sum().clamp_min(1) * cfg.arm_joints)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0 or cfg.smoke:
            print(f"{method} seed={seed} step={step}/{cfg.steps} loss={float(loss.detach()):.6f}", flush=True)
        if step % cfg.eval_freq == 0 or step == cfg.steps:
            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg.to_dict(),
                "method": method,
                "seed": seed,
                "step": step,
                "best_score": best_score,
                "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
                "provenance": expected,
            }
            candidate = directory / "candidate.pt"
            save_checkpoint(candidate, payload)
            # Constructing an evaluation model consumes RNG even in eval mode.
            # Restore it afterwards so resumed and uninterrupted training agree.
            devices = list(range(torch.cuda.device_count())) if device.startswith("cuda") else []
            with torch.random.fork_rng(devices=devices):
                metrics = evaluate(candidate)
            score = (metrics["position_rmse_m"], metrics["rotation_mean_deg"])
            if score < best_score:
                best_score = score
                payload["best_score"] = best_score
                save_checkpoint(directory / "best.pt", payload)
            payload["best_score"] = best_score
            save_checkpoint(last, payload)
            write_json(
                directory / f"development/step{step:07d}.json", {"loss": float(loss.detach()), **metrics}
            )
            print(f"development: {metrics}", flush=True)
            candidate.unlink()
    write_json(
        directory / "complete.json",
        {"steps": cfg.steps, "best_score": best_score, "best_sha256": digest(directory / "best.pt")},
    )

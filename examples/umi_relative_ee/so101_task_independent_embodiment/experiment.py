"""Stage CLI: run with python -m examples.umi_relative_ee.so101_task_independent_embodiment.experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import subprocess
from pathlib import Path

import numpy as np

from .core import (
    CONDITIONS,
    METHODS,
    Config,
    digest,
    extract,
    json_hash,
    manifest,
    motion_from_episode,
    save_npz,
    write_json,
)


def simulator_identity():
    import so101_sim
    from so101_sim.model import packaged_urdf_path

    package = Path(so101_sim.__file__).parent
    return {
        "files": {str(p.relative_to(package)): digest(p) for p in sorted(package.glob("*.py"))},
        "urdf": digest(packaged_urdf_path()),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "piper-sim",
                "mujoco",
                "placo",
                "numpy",
                "scipy",
                "torch",
                "torchvision",
                "datasets",
                "pyarrow",
                "av",
                "matplotlib",
            )
        },
    }


def code_identity():
    directory = Path(__file__).parent
    return {p.name: digest(p) for p in sorted(directory.glob("*.py"))}


def preflight(root, cfg, device, resume):
    import torch

    from .simulation import Simulation

    model_path = Path(cfg.task_checkpoint)
    model_config = json.loads((model_path / "config.json").read_text())
    if not (
        model_config["type"] == "act"
        and model_config["vision_backbone"] == "resnet18"
        and model_config["chunk_size"] == 30
        and model_config.get("use_umi_relative_ee")
        and model_config.get("obs_state_horizon") == 2
        and not model_config.get("use_vae")
        and model_config.get("umi_rot6d_identity_norm")
    ):
        raise ValueError("Expected the frozen ACT-R18-L1, H30, two-pose UMI checkpoint")
    training = json.loads((model_path / "train_config.json").read_text())
    if training["dataset"]["repo_id"].split("/")[-1] != Path(cfg.train_root).name:
        raise ValueError("Task checkpoint training corpus differs from configured corpus")
    inputs = {p.name: digest(p) for p in sorted(model_path.iterdir()) if p.is_file()}
    sizes = []
    for path, expected in ((cfg.train_root, 1459), (cfg.val_root, 100)):
        info = json.loads((Path(path) / "meta/info.json").read_text())
        if info["total_episodes"] != expected or info["fps"] != 30:
            raise ValueError(f"Wrong dataset size/rate: {path}")
        sizes.append(info["total_episodes"])
    sim = Simulation(cfg, "native")
    rng = np.random.default_rng(cfg.seed)
    max_fk = 0.0
    max_rot = 0.0
    from .core import rotation_errors

    for _ in range(3 if cfg.smoke else 20):
        sim.reset(sim.sample_start(rng))
        _, q, _, actual = sim.state()
        fk = sim.ik.fk(q)
        max_fk = max(max_fk, float(np.linalg.norm(actual[:3, 3] - fk[:3, 3])))
        max_rot = max(max_rot, float(rotation_errors(actual[None], fk[None])[0]))
    if max_fk > 1e-5 or max_rot > 0.001:
        raise RuntimeError(f"FK/site mismatch: {max_fk} m, {max_rot} deg")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; full training requires a working host NVIDIA driver")
    if device.startswith("cuda"):
        (torch.ones(8, device=device) @ torch.ones(8, device=device)).item()
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    evidence = {
        "config": cfg.to_dict(),
        "task_checkpoint": inputs,
        "simulator": simulator_identity(),
        "code": code_identity(),
        "git_commit": git,
        "device": device,
        "dataset_episodes": sizes,
        "max_fk_position_error_m": max_fk,
        "max_fk_rotation_error_deg": max_rot,
    }
    manifest(root / "preflight.json", evidence, resume)
    print(json.dumps(evidence, indent=2), flush=True)


def collection_path(root, condition, generation, split, seed):
    if generation == "D1":
        return root / f"collections/{condition}/{generation}_seed{seed}/{split}"
    return root / f"collections/{condition}/{generation}/{split}"


def training_path(root, condition, generation, method, seed):
    return root / f"train/{condition}/{generation}/{method}_seed{seed}"


def collect(root, cfg, condition, generation, split, seed, device, resume):
    from .learning import LearnedController
    from .simulation import Simulation

    if generation not in ("D0", "D1", "extra_bootstrap") or split not in ("train", "dev"):
        raise ValueError("Invalid collection generation or split")
    if generation != "D0" and split != "train":
        raise ValueError("Development data stays fixed at D0")
    source = json.loads((root / "motions/manifest.json").read_text())
    if not (root / "motions/complete.json").exists():
        raise ValueError("Run extraction first")
    directory = collection_path(root, condition, generation, split, seed)
    controller = None
    checkpoint = None
    if generation == "D1":
        checkpoint = training_path(root, condition, "C0", "hindsight", seed) / "best.pt"
        controller = LearnedController(checkpoint, device)
    expected = {
        "config": cfg.to_dict(),
        "motions": digest(root / "motions/manifest.json"),
        "condition": condition,
        "generation": generation,
        "split": split,
        "controller": digest(checkpoint) if checkpoint else "ik",
        "simulator": simulator_identity(),
        "code": code_identity(),
    }
    manifest(directory / "manifest.json", expected, resume)
    if (directory / "complete.json").exists():
        return
    sim = Simulation(cfg, condition)
    ids = source["splits"][split]
    # D1 and its equal-budget bootstrap control use the SAME new motion/start seeds.
    stream = (1 if split == "dev" else 0) + (10 if generation != "D0" else 0)
    count = cfg.dev_rollouts if split == "dev" else cfg.train_rollouts
    rejected_short = []
    eligible = []
    for eid in ids:
        with np.load(root / f"motions/episodes/{eid:06d}.npz") as ep:
            (eligible if ep["times"][-1] - ep["times"][0] > 29 / 30 + 1e-6 else rejected_short).append(eid)
    if not eligible:
        raise ValueError("No source episodes long enough for collection")
    for index in range(count):
        output = directory / f"rollouts/{index:06d}.npz"
        metadata = output.with_suffix(".json")
        if output.exists() and metadata.exists():
            if digest(output) != json.loads(metadata.read_text())["sha256"]:
                raise ValueError(f"Corrupt rollout: {output}")
            continue
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, stream, index]))
        eid = int(rng.choice(eligible))
        with np.load(root / f"motions/episodes/{eid:06d}.npz") as ep:
            motion = motion_from_episode(ep, rng)
        start = sim.sample_start(rng)
        rollout, feasibility = sim.rollout(motion, start, controller, record_nominal=True)
        rollout.update(
            motion_relative=motion["relative"],
            motion_times=motion["times"],
            gripper_query=motion["gripper"],
            source_episode=np.array(eid),
        )
        save_npz(output, **rollout)
        write_json(
            metadata,
            {
                "source_episode": eid,
                "frame": motion["frame"],
                "augmentation": motion["augmentation"],
                "split": split,
                "generation": generation,
                "index": index,
                "sha256": digest(output),
                **feasibility,
            },
        )
        if index % 100 == 0 or cfg.smoke:
            print(f"collect {condition}/{generation}/{split} {index + 1}/{count}", flush=True)
    write_json(
        directory / "complete.json",
        {
            "rollouts": count,
            "excluded_short_episodes": rejected_short,
            "manifest": json_hash(expected),
        },
    )


def train_stage(root, cfg, condition, generation, method, seed, device, resume):
    from .learning import LearnedController, collection_files, prepare, train
    from .simulation import Simulation, rollout_metrics

    if method == "ik":
        raise ValueError("Runtime IK has no training stage")
    if not cfg.smoke and not device.startswith("cuda"):
        raise ValueError("Full training requires --device cuda; CPU is reserved for the smoke preset")
    if generation != "C0" and method != "hindsight":
        raise ValueError("C1 and extra-data controls use hindsight learning")
    sources = [collection_path(root, condition, "D0", "train", seed)]
    if generation in ("C1", "bootstrap20k"):
        sources.append(
            collection_path(root, condition, "D1" if generation == "C1" else "extra_bootstrap", "train", seed)
        )
    if generation not in ("C0", "C1", "bootstrap20k"):
        raise ValueError(generation)
    prepared_name = f"{generation}_seed{seed}" if generation == "C1" else generation
    prepared = prepare(root / f"prepared/{condition}/{prepared_name}/{method}", sources, method, cfg)
    development = collection_path(root, condition, "D0", "dev", seed)
    dev_files = collection_files(development)
    sim = Simulation(cfg, condition)

    def evaluate(checkpoint):
        controller = LearnedController(checkpoint, device)
        values = []
        for file in dev_files:
            with np.load(file) as source:
                motion = {
                    "relative": source["motion_relative"],
                    "times": source["motion_times"],
                    "gripper": source["gripper_query"],
                }
                result, _ = sim.rollout(motion, source["start"], controller)
            result["gripper_query"] = motion["gripper"]
            values.append(rollout_metrics(result, cfg))
        return {key: float(np.mean([v[key] for v in values])) for key in values[0]}

    train(
        training_path(root, condition, generation, method, seed),
        prepared,
        method,
        cfg,
        seed,
        device,
        evaluate,
        resume,
        {
            "manifest": digest(development / "manifest.json"),
            "rollouts": {p.name: digest(p) for p in dev_files},
            "code": code_identity(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "init",
            "preflight",
            "extract",
            "collect",
            "train",
            "cache-act",
            "evaluate",
            "report",
            "all",
        ),
    )
    parser.add_argument("--root", type=Path, default=Path("/mnt/data1/projects/lerobot-embodiment-exp-so101"))
    parser.add_argument("--preset", choices=("full", "smoke"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--condition", choices=CONDITIONS, default="native")
    parser.add_argument("--generation", default="D0")
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--method", choices=METHODS, default="hindsight")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    import torch

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root = args.root
    config_path = root / "config.json"
    if args.stage in ("init", "all"):
        cfg = Config.smoke_config() if args.preset == "smoke" else Config()
        manifest(config_path, cfg.to_dict(), args.resume)
    cfg = Config.read(config_path)
    if args.stage == "init":
        return
    if args.stage == "preflight":
        preflight(root, cfg, args.device, args.resume)
    elif args.stage == "extract":
        extract(root, cfg, args.resume)
    elif args.stage == "collect":
        collect(root, cfg, args.condition, args.generation, args.split, args.seed, args.device, args.resume)
    elif args.stage == "train":
        train_stage(
            root, cfg, args.condition, args.generation, args.method, args.seed, args.device, args.resume
        )
    elif args.stage in ("cache-act", "evaluate", "report"):
        from .evaluation import cache_act, evaluate_stage, report

        if args.stage == "cache-act":
            cache_act(root, cfg, args.device, args.resume)
        elif args.stage == "evaluate":
            evaluate_stage(
                root, cfg, args.condition, args.generation, args.method, args.seed, args.device, args.resume
            )
        else:
            report(root, cfg)
    elif args.stage == "all":
        from .evaluation import cache_act, evaluate_stage, report

        preflight(root, cfg, args.device, args.resume)
        extract(root, cfg, args.resume)
        cache_act(root, cfg, args.device, args.resume)
        seeds = (1000,) if cfg.smoke else (1000, 2000, 3000)
        for condition in CONDITIONS:
            for split in ("train", "dev"):
                collect(root, cfg, condition, "D0", split, 1000, args.device, args.resume)
            collect(root, cfg, condition, "extra_bootstrap", "train", 1000, args.device, args.resume)
            evaluate_stage(root, cfg, condition, "C0", "ik", 1000, args.device, args.resume)
            for seed in seeds:
                for method in ("residual", "query", "hindsight"):
                    train_stage(root, cfg, condition, "C0", method, seed, args.device, args.resume)
                    evaluate_stage(root, cfg, condition, "C0", method, seed, args.device, args.resume)
                collect(root, cfg, condition, "D1", "train", seed, args.device, args.resume)
                for generation in ("C1", "bootstrap20k"):
                    train_stage(root, cfg, condition, generation, "hindsight", seed, args.device, args.resume)
                    evaluate_stage(
                        root, cfg, condition, generation, "hindsight", seed, args.device, args.resume
                    )
        report(root, cfg)


if __name__ == "__main__":
    main()

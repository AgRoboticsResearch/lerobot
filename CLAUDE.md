# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LeRobot is a PyTorch-based robotics library by Hugging Face for imitation learning and reinforcement learning. It provides pretrained models, datasets, simulation environments, and tools for real-world robotics.

## Development Commands

### Installation

For development, use `poetry` or `uv`:

```bash
# Using poetry
poetry sync --all-extras  # Install everything including simulation environments
poetry sync --extra dev --extra test  # Minimal dev setup

# Using uv
uv sync --all-extras
uv sync --extra dev --extra test
```

After pulling changes with updated `pyproject.toml`, run:
```bash
poetry sync  # or: uv sync
```

### Testing

```bash
# Run all tests
python -m pytest -sv ./tests

# Run specific test file
pytest tests/<TEST_TO_RUN>.py

# End-to-end training tests (via Makefile)
make test-end-to-end DEVICE=cpu  # or cuda
```

Note: Install git-lfs and pull test artifacts:
```bash
git lfs install
git lfs pull
```

### Linting and Formatting

```bash
# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files

# Run manually on staged files
pre-commit

# Run specific tools
ruff format .        # Format code
ruff check --fix .   # Lint and auto-fix
```

### Training and Evaluation

```bash
# Train a policy
lerobot-train --policy.type=act --env.type=aloha --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human

# Evaluate a policy
lerobot-eval --policy.path=lerobot/diffusion_pusht --env.type=pusht --eval.n_episodes=10

# Visualize datasets
lerobot-dataset-viz --repo-id lerobot/pusht --episode-index 0
```

## Architecture

### Core Components

**Datasets (`src/lerobot/datasets/`)**
- `LeRobotDataset`: Main dataset class that loads from Hugging Face Hub or local folders
- Stores observations (images, states) and actions as PyTorch tensors
- Uses Hugging Face `datasets` library with Arrow/Parquet backend
- Videos stored as mp4 files to save space
- Supports `delta_timestamps` for temporal queries (e.g., retrieve frames at relative time offsets)
- Metadata (`meta`) contains: info, episodes, stats, tasks

**Policies (`src/lerobot/policies/`)**
- Implemented policies: ACT, Diffusion, TDMPC, VQ-BeT, and others (pi0, smolvla, sac)
- Each policy has three core files:
  - `configuration_<name>.py`: Dataclass config (e.g., `DiffusionPolicyConfig`)
  - `modeling_<name>.py`: The `nn.Module` model class
  - `processor_<name>.py`: Pre/post-processing for normalization/action transformation
- `factory.py` and `pretrained.py`: Policy creation and loading utilities

**Environments (`src/lerobot/envs/`)**
- Gymnasium-based environments for simulation
- Supported: ALOHA, PushT, XArm, Libero
- `make_env()` factory creates vectorized environments (Sync/Async)

**Robots (`src/lerobot/robots/`)**
- Real robot control: Koch, ALOHA, SO100, SO101, LeKiwi, HopeJR
- `robot.py` defines base `Robot` class with `connect()`, `send()`, `disconnect()` pattern
- Cameras and motors are abstracted (see below)

**Hardware Abstractions**
- `cameras/`: OpenCV and Intel RealSense support
- `motors/`: Dynamixel and Feetech servo drivers
- `teleoperators/`: Teleoperation control

**Processor Pipeline (`src/lerobot/processor/`)**
- `PolicyProcessorPipeline`: Chain of processors for input/output transformation
- Common processors: normalization, device placement, delta actions, observation filtering
- Processors are stateful and serializable

**Configs (`src/lerobot/configs/`)**
- `parser.py`: Draccus-based CLI parser with `@parser.wrap()` decorator
- `train.py`, `eval.py`, `policies.py`: Dataclass configs for pipelines
- Hierarchical override via CLI: `--policy.learning_rate=1e-4`

**Training Loop (`src/lerobot/scripts/lerobot_train.py`)**
- Main `train()` function orchestrates the pipeline
- `update_policy()`: Single training step (forward/backward/optimizer step)
- Supports: checkpointing, evaluation, Weights & Biases logging, AMP (mixed precision)
- Uses `EpisodeAwareSampler` for episode-aware batch sampling

### Key Design Patterns

1. **Registration via `lerobot/__init__.py`**: When adding new policies, environments, or datasets, update the lists (`available_policies`, `available_envs`, etc.) in this central registry file.

2. **Factory Pattern**: `make_policy()`, `make_dataset()`, `make_env()`, `make_optimizer_and_scheduler()` create instances from configs.

3. **Pre/Post Processing**: Policies always have paired preprocessor and postprocessor pipelines that handle normalization, device placement, and action transformations.

4. **Config-Driven**: All training/eval parameters are dataclasses in `configs/`, overridden via CLI using Draccus.

5. **Dataset Format**: LeRobotDataset wraps Hugging Face datasets with video support, episode indexing, and temporal querying.

## Adding New Features

### New Policy
1. Create `src/lerobot/policies/<name>/` with `configuration_*.py`, `modeling_*.py`, `processor_*.py`
2. Update `available_policies` and `available_policies_per_env` in `lerobot/__init__.py`
3. Set `name` class attribute in config
4. Import in `tests/test_available.py`

### New Environment
1. Update `available_tasks_per_env` and `available_datasets_per_env` in `lerobot/__init__.py`

### New Dataset
1. Update `available_datasets_per_env` in `lerobot/__init__.py`

## Configuration System

- Use `@parser.wrap()` decorator on main functions for automatic CLI parsing
- Configs are dataclasses; override nested values with dot notation: `--policy.device=cuda`
- Resume training with `--config_path=outputs/.../train_config.json --resume=true`
- Pretrained policies loaded via `--policy.path=<repo_id_or_local_path>`

## Project Structure Notes

- `src/lerobot/`: Main package
- `tests/`: Pytest test suite (mirrors `src/lerobot/` structure)
- `examples/`: Example scripts
- `configs/`: YAML config files referenced in documentation
- `outputs/`: Training outputs (checkpoints, logs, videos)

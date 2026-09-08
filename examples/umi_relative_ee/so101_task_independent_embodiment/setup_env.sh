#!/usr/bin/env bash
# Overlay simulator dependencies in a dedicated environment; reuse the existing LeRobot installation.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
BASE_PYTHON="${UMI_EMB_BASE_PYTHON:-$REPO/.venv/bin/python}"
ENV_DIR="${UMI_EMB_ENV:-/mnt/data1/projects/lerobot-embodiment-env}"
SIM_SOURCE="${PIPER_SIM_SOURCE:-/home/zfei/code/piper/piper_mujoco}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/umi-emb-uv-cache}"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  uv venv --python "$BASE_PYTHON" "$ENV_DIR"
fi
# A .pth file exposes the base installation without upgrading its dependencies.
uv run --no-project --python "$BASE_PYTHON" python - "$ENV_DIR" "$REPO" <<'PY'
import pathlib
import sys
import sysconfig

env, repo = map(pathlib.Path, sys.argv[1:])
site = env / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
site.mkdir(parents=True, exist_ok=True)
(site / "lerobot_workspace.pth").write_text(
    sysconfig.get_paths()["purelib"] + "\n" + str(repo / "src") + "\n" + str(repo) + "\n"
)
PY
uv pip install --python "$ENV_DIR/bin/python" -r "$SCRIPT_DIR/requirements-sim.txt" "$SIM_SOURCE"
uv pip freeze --python "$ENV_DIR/bin/python" > "$ENV_DIR/resolved-requirements.txt"
uv run --no-project --python "$ENV_DIR/bin/python" python -c \
  'import torch, scipy, mujoco, placo, so101_sim; print("Simulator imports OK; CUDA:", torch.cuda.is_available())'
echo "Environment ready: $ENV_DIR"

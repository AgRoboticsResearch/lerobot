#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
ENV_DIR="${UMI_EMB_ENV:-/mnt/data1/projects/lerobot-embodiment-env}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/umi-emb-uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/umi-emb-matplotlib}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/umi-emb-hf-datasets}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Missing experiment environment. Run bash $SCRIPT_DIR/setup_env.sh first." >&2
  exit 2
fi
cd "$REPO"
ARTIFACT_ROOT=/mnt/data1/projects/lerobot-embodiment-exp-so101
READ_ROOT=0
for ARG in "$@"; do
  if [[ "$ARG" == --help || "$ARG" == -h ]]; then
    exec uv run --no-project --python "$ENV_DIR/bin/python" python \
      -m examples.umi_relative_ee.so101_task_independent_embodiment.experiment "$@"
  elif [[ "$READ_ROOT" == 1 ]]; then
    ARTIFACT_ROOT="$ARG"
    READ_ROOT=0
  elif [[ "$ARG" == --root ]]; then
    READ_ROOT=1
  elif [[ "$ARG" == --root=* ]]; then
    ARTIFACT_ROOT="${ARG#--root=}"
  fi
done
mkdir -p "$ARTIFACT_ROOT/logs"
RUN_LOG="$(mktemp "$ARTIFACT_ROOT/logs/run-$(date -u +%Y%m%dT%H%M%S)-XXXXXX.log")"
trap 'RUN_STATUS=$?; printf "%s\n" "$RUN_STATUS" > "$RUN_LOG.exit_code"' EXIT
uv run --no-project --python "$ENV_DIR/bin/python" python \
  -m examples.umi_relative_ee.so101_task_independent_embodiment.experiment "$@" 2>&1 | tee "$RUN_LOG"

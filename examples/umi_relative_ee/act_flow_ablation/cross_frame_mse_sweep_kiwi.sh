#!/usr/bin/env bash
# Full adjacent-frame direct-MSE sweep on kiwi. One prediction pair supplies
# both h10 and h30 metrics, over the canonical 100 episodes x 5 anchors.
set -uo pipefail

REPO=/home/zfei/code/lerobot-fei-v5.0-umi-unified
PY=$REPO/.venv/bin/python
EVAL=$REPO/examples/umi_relative_ee/eval_open_loop_dataset.py
ROOT=/mnt/data/zfei/lerobot-act-flow-ablation/eval/cross_frame_mse
MANIFEST=$ROOT/manifest.tsv
VAL_ROOT=/home/zfei/data/sroiv2_strawberry_picking_lab_validation
VAL_REPO=local/sroiv2_strawberry_picking_lab_validation
MAX_LIGHT_JOBS=${MAX_LIGHT_JOBS:-2}
MAX_VLM_JOBS=${MAX_VLM_JOBS:-1}
ONLY_LIGHT=${ONLY_LIGHT:-0}
SKIP_LIGHT=${SKIP_LIGHT:-0}
VLM_POLICY_FILTER=${VLM_POLICY_FILTER:-}
GATE_MIB=${GATE_MIB:-6500}

mkdir -p "$ROOT/logs" "$ROOT/status"
cd "$REPO"

free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1
}

gate_vram() {
  local required=$1
  while [ "$(free_mib)" -lt "$required" ]; do
    sleep 30
  done
}

run_one() {
  local run=$1 checkpoint=$2 policy_type=$3
  local out=$ROOT/tree/$run/seed1000
  local log=$ROOT/logs/${run}.log
  local status=$ROOT/status/${run}.status
  if compgen -G "$out/*_cross_frame_mse_metrics.json" >/dev/null; then
    printf 'done\t%s\n' "$run" >"$status"
    return 0
  fi
  gate_vram "$GATE_MIB"
  mkdir -p "$out"
  printf '[%s] start %s (%s) free=%s MiB\n' "$(date '+%F %T')" "$run" "$policy_type" "$(free_mib)" >>"$ROOT/driver.log"
  if HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
      PYTHONPATH=src timeout 14400 "$PY" "$EVAL" \
      --pretrained_path="$checkpoint" \
      --dataset_root="$VAL_ROOT" --repo_id="$VAL_REPO" \
      --samples_per_episode=5 \
      --query_min_action_offset=-1 --query_max_action_offset=31 \
      --cross_frame_mse_eval --cross_frame_mse_horizons=10,30 \
      --seed=1000 --device=cuda --video_backend=pyav --output_dir="$out" \
      >"$log" 2>&1; then
    printf 'done\t%s\n' "$run" >"$status"
    printf '[%s] done %s\n' "$(date '+%F %T')" "$run" >>"$ROOT/driver.log"
  else
    rc=$?
    printf 'failed:%s\t%s\n' "$rc" "$run" >"$status"
    printf '[%s] FAILED rc=%s %s\n' "$(date '+%F %T')" "$rc" "$run" >>"$ROOT/driver.log"
    return "$rc"
  fi
}

"$PY" examples/umi_relative_ee/act_flow_ablation/build_cross_frame_mse_manifest.py \
  --output "$MANIFEST" >>"$ROOT/driver.log" 2>&1

# Lightweight ACT/DP rows: at most two beside the active trainer. The 6.5-GiB
# launch gate prevents a second process from starting if the first is larger
# than expected.
if [ "$SKIP_LIGHT" != 1 ]; then
  while IFS=$'\t' read -r run checkpoint policy_type source_report; do
    run_one "$run" "$checkpoint" "$policy_type" &
    while [ "$(jobs -pr | wc -l)" -ge "$MAX_LIGHT_JOBS" ]; do
      wait -n || true
    done
  done < <(awk -F '\t' 'NR > 1 && $3 != "pi0" && $3 != "pi05" && $3 != "smolvla"' "$MANIFEST")
  wait || true
fi

if [ "$ONLY_LIGHT" = 1 ]; then
  done_count=$(grep -l '^done' "$ROOT"/status/*.status 2>/dev/null | wc -l)
  failed_count=$(grep -l '^failed' "$ROOT"/status/*.status 2>/dev/null | wc -l)
  printf '[%s] light-only complete done=%s failed=%s\n' \
    "$(date '+%F %T')" "$done_count" "$failed_count" >>"$ROOT/driver.log"
  exit 0
fi

# π0.5 rows remain serial; low-memory SmolVLA rows may be run by a dedicated
# helper with VLM_POLICY_FILTER=smolvla and MAX_VLM_JOBS=2.
while IFS=$'\t' read -r run checkpoint policy_type source_report; do
  if [ -n "$VLM_POLICY_FILTER" ] && [ "$policy_type" != "$VLM_POLICY_FILTER" ]; then
    continue
  fi
  run_one "$run" "$checkpoint" "$policy_type" &
  while [ "$(jobs -pr | wc -l)" -ge "$MAX_VLM_JOBS" ]; do
    wait -n || true
  done
done < <(awk -F '\t' 'NR > 1 && ($3 == "pi0" || $3 == "pi05" || $3 == "smolvla")' "$MANIFEST")
wait || true

done_count=$(grep -l '^done' "$ROOT"/status/*.status 2>/dev/null | wc -l)
failed_count=$(grep -l '^failed' "$ROOT"/status/*.status 2>/dev/null | wc -l)
printf '[%s] complete done=%s failed=%s\n' "$(date '+%F %T')" "$done_count" "$failed_count" >>"$ROOT/driver.log"

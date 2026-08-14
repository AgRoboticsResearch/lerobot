#!/usr/bin/env bash
# Retrain ALL 14 multi-seed confirmation runs fresh on /mnt/data1 (the external
# artifact disk failed and its checkpoints are stranded). Four companion queues
# cover the 7 variants x {seed 2000, seed 3000} matrix; each companion runs its
# subset sequentially while wait_for_slot gates total concurrency to <=4 with a
# VRAM margin. Launches are VRAM-staggered so jobs allocate memory before the
# next companion's gate check, avoiding a ramp-up OOM race.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UMI_ABLATION_ROOT="${UMI_ABLATION_ROOT:-/mnt/data1/projects/lerobot-arch-exp}"
export UMI_MAX_CONCURRENT="${UMI_MAX_CONCURRENT:-4}"
export UMI_MIN_FREE_VRAM="${UMI_MIN_FREE_VRAM:-5000}"
LAUNCH_LOG="$UMI_ABLATION_ROOT/logs/confirmation_fresh_launch_$(date '+%Y%m%d_%H%M%S').log"
mkdir -p "$UMI_ABLATION_ROOT/logs"
exec > >(tee -a "$LAUNCH_LOG") 2>&1

timestamp() { date '+%F %T'; }
used_vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -1; }
wait_for_vram_climb() {
  local baseline="$1" delta="${2:-3000}" cur waited=0
  while (( waited < 240 )); do
    cur="$(used_vram)"; cur="${cur:-0}"
    (( cur - baseline >= delta )) && { echo "[$(timestamp)] VRAM climbed ${baseline}->${cur}; proceeding"; return 0; }
    sleep 10; waited=$((waited + 10))
  done
  echo "[$(timestamp)] VRAM-climb wait timed out at $(used_vram); proceeding anyway"
}
launch_companion() {
  local seed="$1"; shift
  local sess="$1"; shift
  local before; before="$(used_vram)"; before="${before:-0}"
  tmux new-session -d -s "$sess" \
    "UMI_ABLATION_ROOT=$UMI_ABLATION_ROOT UMI_MAX_CONCURRENT=$UMI_MAX_CONCURRENT UMI_MIN_FREE_VRAM=$UMI_MIN_FREE_VRAM $SCRIPT_DIR/run_companion.sh $seed $*"
  echo "[$(timestamp)] launched $sess: run_companion.sh $seed $* (VRAM before=${before})"
  wait_for_vram_climb "$before" 3000
}

echo "[$(timestamp)] === confirmation FRESH retrain start; root=$UMI_ABLATION_ROOT ==="
echo "[$(timestamp)] initial VRAM used: $(used_vram)"

# Balanced 4-queue split of the 14 runs (R50-heavy queues get fewer variants).
launch_companion 2000 fresh_s2_r50   act_r50_vae act_r50_v1_vae act_r18_vae
launch_companion 2000 fresh_s2_r18   act_r18_l1 act_r18_flow_u_lr1e5 act_r18_diffusion_lr1e5 diffusion_r18
launch_companion 3000 fresh_s3_r50   act_r50_vae act_r50_v1_vae act_r18_vae
launch_companion 3000 fresh_s3_r18   act_r18_l1 act_r18_flow_u_lr1e5 act_r18_diffusion_lr1e5 diffusion_r18

echo "[$(timestamp)] === all 4 companions dispatched; VRAM used: $(used_vram) ==="

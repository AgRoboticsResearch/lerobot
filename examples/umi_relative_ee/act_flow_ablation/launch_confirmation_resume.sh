#!/usr/bin/env bash
# Launch the 5 remaining confirmation runs (4 resume + 1 fresh) as resume-aware
# companion queues, with VRAM-aware staggering so jobs do not pile up during the
# model-load ramp and OOM a resume (which would risk archiving an intact partial
# checkpoint). Each companion self-gates further via wait_for_slot (<=4 jobs,
# MIN_FREE_VRAM margin); this launcher only sequences the FIRST allocation of
# each job to make the gate's count/VRAM observations accurate.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UMI_ABLATION_ROOT="${UMI_ABLATION_ROOT:-/media/zfei/Glowat512/projects/lerobot-arch-exp}"
export UMI_MAX_CONCURRENT="${UMI_MAX_CONCURRENT:-4}"
export UMI_MIN_FREE_VRAM="${UMI_MIN_FREE_VRAM:-5000}"
LAUNCH_LOG="$UMI_ABLATION_ROOT/logs/confirmation_resume_launch_$(date '+%Y%m%d_%H%M%S').log"
mkdir -p "$UMI_ABLATION_ROOT/logs"
exec > >(tee -a "$LAUNCH_LOG") 2>&1

timestamp() { date '+%F %T'; }

used_vram() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -1
}

# Wait until VRAM-used has climbed by at least DELTA MiB since `baseline`, proving
# the just-launched job has actually allocated memory (not merely registered a
# process). Caps the wait so a stuck launch does not block the whole queue.
wait_for_vram_climb() {
  local baseline="$1" delta="${2:-3000}" cur waited=0
  while (( waited < 240 )); do
    cur="$(used_vram)"; cur="${cur:-0}"
    if (( cur - baseline >= delta )); then
      echo "[$(timestamp)] VRAM climbed ${baseline}->${cur} MiB (>=${delta}); proceeding"
      return 0
    fi
    sleep 10
    waited=$((waited + 10))
  done
  cur="$(used_vram)"; cur="${cur:-0}"
  echo "[$(timestamp)] VRAM-climb wait timed out at ${cur} MiB (baseline ${baseline}); proceeding anyway"
}

launch_companion() {
  local seed="$1" variant="$2" sess="$3" before after
  before="$(used_vram)"; before="${before:-0}"
  tmux new-session -d -s "$sess" \
    "UMI_ABLATION_ROOT=$UMI_ABLATION_ROOT UMI_MAX_CONCURRENT=$UMI_MAX_CONCURRENT UMI_MIN_FREE_VRAM=$UMI_MIN_FREE_VRAM $SCRIPT_DIR/run_companion.sh $seed $variant"
  echo "[$(timestamp)] launched $sess: run_companion.sh $seed $variant (VRAM before=${before} MiB)"
  wait_for_vram_climb "$before" 3000
}

echo "[$(timestamp)] === confirmation-resume launcher start; root=$UMI_ABLATION_ROOT ==="
echo "[$(timestamp)] initial VRAM used: $(used_vram) MiB"

# Order: longest/heaviest first so the gate observations are stable.
#   - act_r50_v1_vae s3000: FRESH 100k (longest)
#   - act_r50_vae     s3000: resume @60k (R50)
#   - act_r50_v1_vae  s2000: resume @50k (R50)
#   - act_r18_flow    s3000: resume @30k (R18, lighter -> 4th slot)
#   - diffusion_r18   s3000: resume @30k (self-gates as the 5th)
launch_companion 3000 act_r50_v1_vae        conf_r50v1_s3000
launch_companion 3000 act_r50_vae           conf_r50vae_s3000
launch_companion 2000 act_r50_v1_vae        conf_r50v1_s2000
launch_companion 3000 act_r18_flow_u_lr1e5  conf_flow_s3000
# The 5th may immediately wait_for_slot if 4 are already running; that is correct.
launch_companion 3000 diffusion_r18         conf_diff_s3000

echo "[$(timestamp)] === all 5 companions dispatched; final VRAM used: $(used_vram) MiB ==="
echo "[$(timestamp)] monitor with: tmux ls ; tail -f $UMI_ABLATION_ROOT/logs/companion_seed*.log"

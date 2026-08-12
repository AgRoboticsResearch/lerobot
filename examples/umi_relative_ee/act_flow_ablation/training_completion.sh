#!/usr/bin/env bash
# Shared durable-completion checks for fault-tolerant experiment supervisors.
# A final checkpoint alone is insufficient: require the trainer's terminal log
# message and the complete resumable state before recovering a wrapper anomaly.

durable_training_checkpoint() {
  local artifact_root="$1" run_name="$2" steps="$3" checkpoint padded_steps step_file
  checkpoint="$artifact_root/train/$run_name/checkpoints/$steps"
  if [[ ! -d "$checkpoint" ]]; then
    printf -v padded_steps '%06d' "$steps"
    checkpoint="$artifact_root/train/$run_name/checkpoints/$padded_steps"
  fi
  step_file="$checkpoint/training_state/training_step.json"

  [[ -s "$checkpoint/pretrained_model/model.safetensors" ]] &&
    [[ -s "$checkpoint/pretrained_model/config.json" ]] &&
    [[ -s "$checkpoint/pretrained_model/train_config.json" ]] &&
    [[ -s "$checkpoint/pretrained_model/policy_preprocessor.json" ]] &&
    [[ -s "$checkpoint/pretrained_model/policy_postprocessor.json" ]] &&
    [[ -s "$checkpoint/training_state/optimizer_state.safetensors" ]] &&
    [[ -s "$checkpoint/training_state/optimizer_param_groups.json" ]] &&
    [[ -s "$checkpoint/training_state/rng_state.safetensors" ]] &&
    [[ -s "$step_file" ]] &&
    grep -Eq '"step"[[:space:]]*:[[:space:]]*'"$steps"'([[:space:]]*[,}])' < <(tr -d '\r\n' < "$step_file")
}

training_is_complete() {
  local artifact_root="$1" run_name="$2" steps="$3" log
  log="$artifact_root/logs/$run_name.log"
  durable_training_checkpoint "$artifact_root" "$run_name" "$steps" &&
    { grep -Fq "] completed $run_name" "$log" || grep -Fq "End of training" "$log"; }
}

recover_training_completion() {
  local artifact_root="$1" run_name="$2" steps="$3" log timestamp
  log="$artifact_root/logs/$run_name.log"
  training_is_complete "$artifact_root" "$run_name" "$steps" || return 1
  if ! grep -Fq "] completed $run_name" "$log"; then
    timestamp="$(date '+%F %T')"
    echo "[$timestamp] recovered-complete: trainer reached End of training and exact final resumable checkpoint passed structural validation" | tee -a "$log"
    echo "[$timestamp] completed $run_name" | tee -a "$log"
  fi
}

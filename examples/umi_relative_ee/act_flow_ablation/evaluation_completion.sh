#!/usr/bin/env bash
# Shared exact-checkpoint predicate for canonical evaluation artifacts.

canonical_evaluation_report() {
  local out="$1" steps="$2" padded reports
  printf -v padded '%06d' "$steps"
  mapfile -t reports < <(
    find "$out" -maxdepth 1 -type f -name "*_${padded}_open_loop_metrics.json" -size +0c 2>/dev/null
  )
  [[ "${#reports[@]}" -eq 1 ]] || return 1
  printf '%s\n' "${reports[0]}"
}

canonical_evaluation_complete() {
  local out="$1" log="$2" run_name="$3" seed="$4" steps="$5"
  canonical_evaluation_report "$out" "$steps" >/dev/null &&
    grep -Fq "] completed evaluation $run_name seed $seed" "$log"
}

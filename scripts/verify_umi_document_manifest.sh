#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
manifest="$repo_root/docs/umi_migration_manifest.md"
paths="$repo_root/docs/umi_migration_paths.tsv"

while IFS=$'\t' read -r source_path destinations disposition; do
  primary_destination=${destinations%%;*}
  if ! git -C "$repo_root" cat-file -e "fei:$source_path"; then
    echo "missing source on fei: $source_path" >&2
    exit 1
  fi
  if [[ ! -e "$repo_root/$primary_destination" ]]; then
    echo "missing migrated destination: $primary_destination" >&2
    exit 1
  fi
  source_checksum=$(git -C "$repo_root" show "fei:$source_path" | sha256sum | awk '{print $1}')
  if ! rg -q "$source_checksum" "$manifest"; then
    echo "manifest checksum missing for: $source_path" >&2
    exit 1
  fi
done < "$paths"

for required in \
  examples/umi_relative_ee/scaling/SCALING_REPORT.pdf \
  examples/umi_relative_ee/prediction_visualization.md \
  docs/legacy/fei-v5.0/prediction_visualization.md \
  docs/legacy/fei-v5.0/pi05_umi_README.md; do
  if [[ ! -e "$repo_root/$required" ]]; then
    echo "missing required v5/working-tree document: $required" >&2
    exit 1
  fi
done

for archived in AGENTS.md CLAUDE.md; do
  if ! cmp -s <(git -C "$repo_root" show "fei:$archived") \
    "$repo_root/docs/legacy/fei-branch/instructions/$archived"; then
    echo "verbatim archive differs: $archived" >&2
    exit 1
  fi
done

prediction_archive_checksum=$(sha256sum "$repo_root/docs/legacy/fei-v5.0/prediction_visualization.md" | awk '{print $1}')
if [[ "$prediction_archive_checksum" != "4c2fcf66a101e762bad12744a25d263b0791aadf6ace8136dd5c7cb3a5f1e5a8" ]]; then
  echo "verbatim archive differs: v5 prediction_visualization.md" >&2
  exit 1
fi

count=$(wc -l < "$paths")
echo "Verified $count tracked fei documents/assets plus PDF, active+archived v5 visualization note, and v5 README archive."

from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path("examples/umi_relative_ee/act_flow_ablation/prefetch_lingbot_assets.sh").resolve()
VALIDATION = Path("examples/umi_relative_ee/act_flow_ablation/lingbot_asset_validation.sh").resolve()


def test_prefetch_validates_both_asset_groups(tmp_path: Path) -> None:
    fake_hf = tmp_path / "fake_hf"
    fake_hf.write_text(
        """#!/usr/bin/env bash
set -eu
repo="$2"
shift 2
local_dir=""
while [[ "$#" -gt 0 ]]; do
  if [[ "$1" == --local-dir ]]; then local_dir="$2"; shift 2; else shift; fi
done
mkdir -p "$local_dir"
if [[ "$repo" == lerobot/lingbot_va_libero_long ]]; then
  printf config > "$local_dir/config.json"
  printf processor > "$local_dir/policy_preprocessor.json"
  printf processor > "$local_dir/policy_postprocessor.json"
  truncate -s 9000000001 "$local_dir/model.safetensors"
else
  mkdir -p "$local_dir/vae" "$local_dir/text_encoder" "$local_dir/tokenizer"
  printf config > "$local_dir/vae/config.json"
  truncate -s 1048577 "$local_dir/vae/diffusion_pytorch_model.safetensors"
  printf config > "$local_dir/text_encoder/config.json"
  truncate -s 1048577 "$local_dir/text_encoder/model.safetensors"
  printf tokenizer > "$local_dir/tokenizer/tokenizer.json"
fi
"""
    )
    fake_hf.chmod(0o755)
    artifact_root = tmp_path / "artifacts"
    env = {
        **os.environ,
        "UMI_ABLATION_ROOT": str(artifact_root),
        "UMI_HF_BIN": str(fake_hf),
        "UMI_LINGBOT_RETRY_SECONDS": "0",
    }

    result = subprocess.run(
        ["bash", str(SCRIPT)], text=True, capture_output=True, env=env, timeout=10, check=False
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "verified trainable LingBot checkpoint" in result.stdout
    assert "verified frozen VAE, text encoder, and tokenizer assets" in result.stdout


def test_shared_validation_rejects_incomplete_shard(tmp_path: Path) -> None:
    trainable = tmp_path / "trainable"
    frozen = tmp_path / "frozen"
    for directory in (trainable, frozen / "vae", frozen / "text_encoder", frozen / "tokenizer"):
        directory.mkdir(parents=True)
    (trainable / "config.json").write_text("config")
    (trainable / "policy_preprocessor.json").write_text("processor")
    (trainable / "policy_postprocessor.json").write_text("processor")
    (trainable / "model.safetensors").write_bytes(b"")
    (trainable / "model.safetensors").touch()
    os.truncate(trainable / "model.safetensors", 9_000_000_001)
    (frozen / "vae" / "config.json").write_text("config")
    (frozen / "vae" / "model.safetensors").write_bytes(b"")
    os.truncate(frozen / "vae" / "model.safetensors", 1_048_577)
    (frozen / "text_encoder" / "config.json").write_text("config")
    (frozen / "text_encoder" / "model.safetensors").write_bytes(b"")
    os.truncate(frozen / "text_encoder" / "model.safetensors", 1_048_577)
    (frozen / "tokenizer" / "tokenizer.json").write_text("tokenizer")

    command = f'. "{VALIDATION}"; lingbot_assets_complete "{trainable}" "{frozen}"'
    assert subprocess.run(["bash", "-c", command], check=False).returncode == 0

    incomplete = frozen / ".cache" / "huggingface" / "download" / "model.incomplete"
    incomplete.parent.mkdir(parents=True)
    incomplete.write_text("partial")
    assert subprocess.run(["bash", "-c", command], check=False).returncode != 0

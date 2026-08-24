# Reproducibility bundle (§9.2.9 / §9.2.13 and the report's raw evidence)

Repository-tracked compact evidence backing the unified-horizon tables and
the physical-dynamics re-evaluation in `../RESEARCH_REPORT.md`. Large raw
artifacts stay external (roots listed at the bottom); everything needed to
verify that those artifacts are the ones the report describes — and to
re-run the numbers — is here.

## Layout

- `query_frames_h10_seed1000.json` — the immutable canonical query list:
  500 (episode_index, frame_index) queries (100 episodes × 5, linspace,
  per-episode bounds [-1, 31], inference seed 1000). sha256
  `6dcb2888fe4f88e7…` — verified identical across host, kiwi, and every
  archived eval tree. Regenerated deterministically by
  `../eval_open_loop_dataset.py` under the §9.2.9 flags.
- `per_episode/` — 88 compact per-run files (`<RUN>.json.gz`): episode-
  balanced means, 95% bootstrap CIs (10k resamples, seed 0), per-episode
  values for every metric (legacy + physical), checkpoint path, and the
  protocol block. Written by `../compile_physical_jerk.py`.
- `per_episode_salvage/` — same format for the 28 recovered runs
  re-scored 2026-08-24 (§9.2.15), from the salvage eval tree
  `eval_salvage_h10/` (listed under external roots). One row's checkpoint
  differs from its run-name budget: `act_r50_v1_vae_seed3000` was scored
  at 20k because its 30k `model.safetensors` is a torn write from the
  disk failure.
- `configs/` — per-run training configuration (`<RUN>.config.json`) copied
  verbatim from each archived checkpoint (draccus-saved). The 88 names map
  1:1 to the §9.2.9 tree runs; `per_episode/*.json.gz` record the exact
  checkpoint each row was scored against.
- `datasets/` — dataset content hashes produced by `hash_dataset.py`
  (also tracked here): meta/tasks/parquet files sha256-hashed individually;
  videos manifest-digested (path, size, mtime_ns) unless noted `--full`.
  `eval_validation_{host,kiwi}.json` are `--full` hashes of the validation
  set and are bit-identical between the two machines (single 159 MB video
  included). Training sets: 1459 rot6d/rotvec (v2.1 reshard; videos live
  outside the dataset root — manifest in
  `train_sroiv2_v21_videos_manifest.json`), 1459/1302 occlusion,
  1000onesb-1125, and the raw lerobot-sroi-v2 root.
- `env/` — interpreter environment freezes for every environment that
  produced a result: host py312 eval env, host uv repo env, host openpi
  venv, kiwi venv (uv-managed venvs have no pip; freezes enumerate
  importlib.metadata distributions), plus the repo `uv.lock` sha256.
- `git_commits.json` — exact code identities: lerobot host branch/commit/
  dirty files; lerobot kiwi (rsync copy, no `.git`) pinned by full
  `rsync --checksum` equivalence to a host commit with the single
  divergence (a training-schedule-only scheduler flag) documented; openpi
  host commit. Kiwi's openpi checkout was deleted after §9.2.5; its
  checkpoints live in the archive below.
- `report_ckpts_sha256_manifest.txt` — full sha256 manifest of the kiwi
  checkpoint archive at the time of §9.2.13 (119 G, 92 runs incl. the 4 JAX
  openpi runs; the manifest predates the 2026-08-24 Glowat512 salvage, which
  added 28 more runs / ~89 G + a separate
  `disk_salvage_glowat512_20260824/manifest_salvage_glowat512.txt`): one
  `<sha256>  <size>  <relpath>` line per file; the file's own sha256 pins
  the manifest.

## Re-running the §9.2.13 numbers

```bash
# 1. re-evaluate a checkpoint (torch policies; any machine with the dataset):
uv run examples/umi_relative_ee/eval_open_loop_dataset.py \
  --policy.path <CKPT>/pretrained_model \
  --dataset.path <sroiv2_strawberry_picking_lab_validation> \
  --samples_per_episode 5 --query_min_action_offset -1 \
  --query_max_action_offset 31 --eval_horizon 10 --seed 1000 --device cuda
# (sweep driver: manifest-driven, idempotent — jerk_sweep_kiwi.sh on kiwi)

# 2. compile + cross-validate against the archived §9.2.9 tree:
uv run examples/umi_relative_ee/act_flow_ablation/compile_physical_jerk.py

# 3. figures:
/home/zfei/anaconda3/envs/py312/bin/python \
  examples/umi_relative_ee/act_flow_ablation/plot_physical_jerk.py

# 4. §9.2.15 (recovered runs) — same compiler over the salvage tree
#    (--no_openpi_carry: those 4 rows are not part of this row set):
uv run examples/umi_relative_ee/act_flow_ablation/compile_physical_jerk.py \
  --jerk_root /mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/eval_salvage_h10 \
  --unified_root /mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/eval_unified_h10 \
  --out_dir /mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/results_salvage_h10 \
  --per_episode_dir examples/umi_relative_ee/act_flow_ablation/repro/per_episode_salvage \
  --no_openpi_carry
/home/zfei/anaconda3/envs/py312/bin/python \
  examples/umi_relative_ee/act_flow_ablation/plot_salvage_h10.py
```

Protocol invariants enforced at compile time: query bounds exactly
{-1, 31}, `eval_horizon` 10, 500 samples, `control_fps` 30, and max |Δ| of
every legacy metric vs the archived tree below 0.1%.

Note on the π0.5-port openpi-recipe 20k row: its archived checkpoint saved
a host-only `scheduler_auto_scale` config key that the kiwi (pre-flag)
LeRobot build rejects at load; the sweep scored a shadow copy — identical
files, config.json with that single key removed (training-schedule-only,
inference-inert). The archive is untouched; the shadow lives at
`patched_ckpts/` next to the kiwi sweep driver.

## External artifact roots (not in the repository)

- Eval trees: `/mnt/data1/projects/lerobot-arch-exp/reeval_v2metrics/`
  (`eval_unified_h10/` archived §9.2.9 tree, `eval_unified_h10_jerk/`
  §9.2.13 re-eval tree, `eval_salvage_h10/` §9.2.15 recovered-runs tree
  (+ its `manifest.tsv` of run → final-checkpoint mappings, incl. the
  s3000@20k torn-ckpt substitution), `eval_common_h32/` §9.2.11, results
  CSVs incl. `results_salvage_h10/`).
- Checkpoint archive (only copy): kiwi `/mnt/data/zfei/archive/report_ckpts/`
  (ssh port 2203) — 119 G / 92 runs at §9.2.13 time, since grown to ~208 G by
  the 2026-08-24 Glowat512 salvage (28 recovered runs folded into the same
  layout; their sha256 manifest + the non-train evidence live in the adjacent
  `disk_salvage_glowat512_20260824/`). Kiwi `/mnt/data/zfei/` layout since
  2026-08-24: `archive/` (report_ckpts + salvage evidence), `eval/`
  (jerk_reeval + salvage_eval sweep trees), `lingbot/` (assets + smoke +
  run_one), `viz/`, `scripts/`.
- Datasets: validation + training roots under `/mnt/data1/sroi/…` (host)
  and `/home/zfei/data/…` (kiwi); hashes in `datasets/`.
- The §9.2.14 openpi h30 bs4 1M training run:
  `/mnt/data1/code/openpi_checkpoints/pi05_lora_sroi_rot6d_h30_bs4_1m/`.

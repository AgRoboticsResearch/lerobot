# Prediction visualization for UMI relative-EE policies (π0.5 / ACT / SmolVLA)

How the open-loop prediction videos are produced for the strawberry-picking
models, so the predicted gripper motion can be compared to ground truth on the
camera image. This document describes the shared method implemented by
`visualize_predictions.py` and doubles as its **run guide** (commands, flags,
output paths). The preserved policy-specific notes remain in
`visualize_predictions_act_smolvla.md` and `../../docs/legacy/fei-v5.0/pi05_umi_README.md`.

For raw unnormalized 9D pose plots and geodesic SO(3) rotation-error analysis,
see `debug_rotation_visualization.md`. Both scripts write under
`<output_dir>/<repo_id>/`, so passing the same output directory to each
co-locates the video, metrics, rotation PNGs, and CSV in one folder.

## What it produces
Per dataset episode, an MP4 where each frame overlays the **predicted** and **GT**
gripper-tip trajectory on the camera image (predicted = green→red gradient, GT =
cyan/yellow), plus a 3D trajectory panel. Used to qualitatively check a model's
predictions on validation or training data.

Output path: `<output_dir>/<repo_id>/pred_episode_<N>.mp4` (default
`outputs/debug/viz_umi`; override with `--output_dir`). Pass `--project` to also
draw the predicted (green→red) and GT (cyan) gripper-tip trajectories on the
camera image; omit it for the panel composite only (no calibration needed).

## Why open-loop (no environment)
UMI-style data has no robot/sim env to roll out in. So instead of a closed-loop
rollout, we condition the policy on the **recorded** observation at each frame,
predict the action chunk, and compare it to the GT motion — an open-loop
prediction check.

## Per-frame pipeline
1. **Load frame with the action delta** `action[t-1, t, t+1, …, t+chunk-1]`
   (built via `resolve_delta_timestamps(policy.config, meta)`; the `[t-1]` is
   needed for state derivation).
2. **Preprocess** through the policy's saved processor:
   - derive the 20D two-pose state from `action[t-1], action[t]` (no state column
     on disk);
   - convert absolute 7D action chunk → relative 10D rot6d, relative to `action[t]`;
   - normalize (QUANTILES, stats from the checkpoint's normalizer);
   - tokenize the language task (π0.5 / SmolVLA are language-conditioned).
3. **`predict_action_chunk`** → predicted relative 10D chunk (normalized).
4. **Unnormalize** → relative 10D actions in real units.
5. **Trajectory**: each chunk target → `rot6d_to_matrix` → 4×4 relative transform;
   positions live in the chunk-start (`action[t]`) frame.
6. **Project** onto the camera image via the D405 hand-eye + intrinsics.
7. **Render** the composite frame; collect frames → MP4.

## The projection (SROI v2 D405 rig)
Hand-eye from `camera_gripper_extrinsics_sroi_v2_d405.json` (copied next to each
script — **no dependency on the `sroi_rosbag_utilities` folder**):
- `T_opt_cam` (optical→camera), `T_cam_ee` (camera→gripper-tip).
- For a relative target `T_rel`:
  `p_tip = (T_opt_cam @ T_rel @ T_cam_ee)[:3, 3]` — the gripper-tip in the current
  camera optical frame.
- Pinhole with the color intrinsics `K`: `px = fx·x/z + cx`, `py = fy·y/z + cy`;
  masked to NaN when behind the camera.
- The chunk-start tip `(T_opt_cam @ T_cam_ee)[:3, 3]` projects to a fixed
  **(327, 321)** — identical to `sroi_rosbag_utilities`'s `visualize_traj_video.py
  --extrinsics-config …sroi_v2_d405.json`. This is the reference the projections
  are matched against.

> ⚠️ Use the **D405 JSON**, not the Piper URDF. The URDF's `camera_link→ee_link`
> transform differs from the measured rig and shifts the start to (355, 229).
> The preserved legacy tools may still use URDF/Placo; the maintained unified
> visualizer uses this JSON via `--extrinsics_config`.

## The unified script
| script (repo / env) | models | output | notes |
|---------------------|--------|--------|-------|
| `examples/umi_relative_ee/visualize_predictions.py` (py312) | ACT, SmolVLA, π0.5/LoRA | `pred_episode_<N>.mp4` | panel composite (camera + 3D + per-dim curves); on-image projection with `--project` |

The script includes the SROI projection helpers (`load_tip_kin`, `project_future`,
`_green_red_gradient`) and the D405 JSON locally. **LoRA adapters are detected
automatically** (if `adapter_config.json` is present it loads the base named there
and applies the adapter), so the same command works for full-fine-tuned ACT /
SmolVLA and for π0.5 LoRA.

## Model-specific notes
- **π0.5 (LoRA)**: load via `PeftConfig` → `from_pretrained(base, config=policy_config)`
  → `PeftModel.from_pretrained(adapter)`. The `config=policy_config` is essential —
  without it the model uses base π0.5's `base_0_rgb` image keys instead of
  `observation.images.camera`. Drive inference through the **dataloader +
  `lerobot_collate_fn`** path (a hand-built batch dict doesn't survive the
  processor's transition mapping; `validate_policy` is the template). Task
  `"pick the strawberry"` is injected for PaliGemma. Point `--pretrained_path`
  at the LoRA adapter dir; the base model named in `adapter_config.json` is loaded
  automatically — no extra flags.
- **ACT**: at inference, pop the `None` ACTION the preprocessor leaves
  (`if processed.get(ACTION) is None: processed.pop(ACTION, None)`), else the VAE
  runs the posterior branch and crashes — inference must use the prior. No `--task`
  needed; best-val checkpoint usually fits tightest.
- **SmolVLA**: needs `--task "pick the strawberry"` (the default, so you can
  usually omit it). PaliGemma-tokenized internally.

## Checkpoints
- **π0.5** — `outputs/train/pi05_lora_umi_relative_ee/checkpoints/`: loss-fix rerun
  best `040000` (val 0.0571), final `050000` (0.0577).
- **SmolVLA** — `outputs/train/smolvla_relative_ee_chunk30_strawberry_1M/checkpoints/`:
  loss-fix rerun `0250000`, `0500000` (training ongoing toward 1M).
- **ACT** — `outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30_sroi_v2_masked_1012train_100val/checkpoints/`:
  best `0200000` (~180k), final `2500000`.

Use **best-val** for the tightest predictions; **final** for the fully-trained view
(both overfit late).

## Validation vs train data
- Validation: `/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation` (100 ep).
- Train: `/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb` (1012 ep).

Swap `--dataset_root` (and `--camera_info_path` when auto-discovery is unavailable). Train
predictions are expected to track GT more tightly (the model saw them).

## Overfit / training-episode check
To confirm the pipeline can fit a specific episode (e.g. after a one-episode
overfit run), point `--dataset_root` at the **training** set and select that
episode. A correct overfit reproduces the GT trajectory near-perfectly (predicted
overlays GT):
```bash
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path outputs/train/smolvla_one_ep_debug/checkpoints/050000/pretrained_model \
  --dataset_root /path/to/sroiv2_strawberry_picking_lab_1000onesb_1125 \
  --episode_indices 0 --project
```

## Requirements
- **Intrinsics K**: any `camera_info_color.json` under the dataset's
  `meta/camera_info/` (D405 is fixed-calibration, so any episode's file works).
- **Hand-eye**: `camera_gripper_extrinsics_sroi_v2_d405.json` next to the script.
- **Environment**: any lerobot env with the policy plus `imageio` / `matplotlib` /
  `cv2`. Locally: `/home/zfei/anaconda3/envs/py312/bin/python`. On kiwi: the unified
  checkout's `.venv/bin/python` (run `uv pip install matplotlib imageio
  imageio-ffmpeg` if missing — `cv2`/`av` come from the `dataset`/`av-dep` extras).
- Running the viz while a training job is active is fine as long as there's GPU
  headroom (~5–6 GB for eval; check `nvidia-smi` first to avoid OOM-ing the run).

## Quick reference (commands)
Minimal run — panel composite only (no calibration needed):
```bash
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path <CKPT>/pretrained_model \
  --dataset_root <DATASET_ROOT> \
  --episode_indices 0 1 2
```
Panel composite **+ on-image trajectory projection** (`--project`):
```bash
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path <CKPT>/pretrained_model \
  --dataset_root <DATASET_ROOT> \
  --episode_indices 0 1 2 --project
```

π0.5 on validation (unified worktree, py312). Video and rotation diagnostics are
co-located by passing the same directory to both scripts (`visualize_predictions.py`
takes `--output_dir`; the rotation script takes `--output-dir`); everything lands
under `<OUT>/<repo_id>/`:
```bash
# projected-trajectory video + prediction_metrics.json
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path outputs/train/pi05_lora_umi_relative_ee/checkpoints/040000/pretrained_model \
  --dataset_root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --episode_indices 0 1 2 --project \
  --output_dir outputs/debug/viz_pi05

# raw-9D + SO(3) rotation diagnostics, into the same folder
/home/zfei/anaconda3/envs/py312/bin/python \
  examples/umi_relative_ee/generate_rotation_comparison.py \
  --visualizer examples/umi_relative_ee/visualize_predictions.py \
  --checkpoint outputs/train/pi05_lora_umi_relative_ee/checkpoints/040000/pretrained_model \
  --dataset-root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --episodes 0 1 2 \
  --output-dir outputs/debug/viz_pi05
```
SmolVLA on validation (same CLI and environment):
```bash
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/visualize_predictions.py \
  --dataset_root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --episode_indices 0 1 2 \
  --pretrained_path outputs/train/smolvla_relative_ee_chunk30_strawberry_1M/checkpoints/0500000/pretrained_model \
  --task "pick the strawberry" \
  --camera_info_path /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation/meta/camera_info/validation_20260714_160922-png__episode_018/camera_info_color.json \
  --output_dir outputs/debug/viz_smolvla
```
(ACT uses the same command with an ACT checkpoint; `--task` is harmless and may be omitted.)

## Options
`--no_gt`, `--task`, `--fps`, `--device`, `--seed`, `--repo_id`, `--output_dir`,
`--extrinsics_config`, `--camera_info_path`, `--max_frames_per_episode`,
`--first_frame_debug`, `--project`, `--num_steps`, `--zero_noise`.

## Zero-noise decoding (`--zero_noise`)
Flow policies (SmolVLA, π0.5) normally seed the flow integrator with a random
Gaussian latent. `--zero_noise` feeds an **all-zero latent** instead — the
"zero all dims" arm of the within-chunk-jitter audit
([`within_chunk_jitter_analysis.md`](./within_chunk_jitter_analysis.md)), which
collapses SmolVLA's rotation-jitter proxy from ~1.9° to ~0.085° and improves its
rotation endpoint from ~4.4° to ~2.6°. It is ignored for non-flow policies (ACT).
The latent shape is derived from the checkpoint config
(`(1, chunk_size, max_action_dim)`). The production sampler integrates the
complete latent width; zero-noise remains a diagnostic only. Diagnostics
retain `--legacy_full_action_noise` for report compatibility, but it no longer
changes SmolVLA/π0.5 inference. See the historical
[`run_jerk_ab.sh`](./run_jerk_ab.sh) and the
zero-noise video runner [`run_zero_noise_vis.sh`](./run_zero_noise_vis.sh).

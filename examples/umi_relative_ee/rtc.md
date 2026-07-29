# RTC with UMI Relative End-Effector Policies

This note describes Real-Time Chunking (RTC) support for Pi0.5 and SmolVLA checkpoints trained with the UMI relative end-effector pipeline in this directory.

## Current compatibility

| Policy and action representation | Current RTC status |
| --- | --- |
| Pi0, Pi0.5, or SmolVLA with absolute actions | Supported |
| Ordinary relative joint actions using `RelativeActionsProcessorStep` | Supported; the RTC rollout engine re-anchors the leftover prefix |
| Pi0.5 with `use_umi_relative_ee=true` | Supported through `RTCInferenceEngine` |
| SmolVLA with `use_umi_relative_ee=true` | Supported through `RTCInferenceEngine` |

Existing UMI checkpoints and LoRA adapters do not require retraining. The support is entirely in the inference-time processor and RTC queue path.

## Why re-anchoring is required

RTC guides the start of a newly generated chunk toward the unexecuted portion of the previous chunk. This requires both chunks to use the same action representation and reference frame.

UMI relative-EE preprocessing expresses every target in a chunk relative to the end-effector pose at the start of that chunk. For the same absolute target:

- A leftover action from the old chunk is represented as `T_old_base^-1 @ T_target`.
- The corresponding action in the new chunk must be represented as `T_current_base^-1 @ T_target`.

These relative values differ whenever the robot has moved between chunk starts. Comparing them directly does not preserve the same absolute target.

The RTC action queue keeps two synchronized representations:

- model-space actions used by the flow-matching denoiser;
- postprocessed absolute 7D actions used for robot execution.

For a new UMI chunk, `RTCInferenceEngine` detects `UmiRelativeActionsStep`, reads the unexecuted absolute actions, and rebuilds the model-space prefix using the latest cached EE pose. It then applies the checkpoint's action normalization before passing the prefix to Pi0.5 or SmolVLA.

Pi0.5 and SmolVLA retain their existing native RTC denoising hooks. Those hooks receive the corrected 10D normalized UMI prefix and do not need policy-specific SE(3) code.

## Recommended rollout path

Use the standard rollout RTC backend and CUDA:

```bash
CUDA_VISIBLE_DEVICES=0 lerobot-rollout \
  --strategy.type=base \
  --policy.path=/path/to/umi_checkpoint/pretrained_model \
  --device=cuda \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=10 \
  --inference.rtc.max_guidance_weight=10.0 \
  --robot.type=<robot_type> \
  <robot and camera options> \
  --task="pick the strawberry" \
  --duration=120
```

Start with the policy defaults of 10 flow-matching denoising steps, `execution_horizon=10`, and `max_guidance_weight=10.0`. Measure actual inference latency and inspect motion continuity before tuning the horizon or guidance weight.

## Behavior boundaries

- Selecting `--inference.type=sync` uses the existing synchronous path and does not run any RTC re-anchoring code.
- If `rtc_config.enabled` is false, the UMI RTC conversion is skipped.
- Training, validation, checkpoint serialization, and ordinary `predict_action_chunk()` calls are unchanged.
- The existing `deploy_umi_relative_ee_piper.py` custom loop does not automatically use `RTCInferenceEngine`.
- A custom direct call to `policy.predict_action_chunk(prev_chunk_left_over=...)` must not pass a stale UMI prefix from an older chunk frame. Custom RTC code must perform the same absolute-to-current-frame conversion described below.

## Implementation flow

Before applying RTC guidance, the rollout inference path:

1. Read the leftover actions in absolute coordinates from the postprocessed action queue.
2. Re-express every leftover end-effector target relative to the latest UMI chunk-start pose using SE(3) composition.
3. Convert the relative rotations to the row-based 6D rotation representation used by the UMI training pipeline.
4. Normalize the re-anchored actions using the checkpoint's training statistics.
5. Pass this new model-space prefix to the Pi0.5 or SmolVLA RTC denoising hook.

The queue continues to keep both representations so robot execution remains absolute while future guidance can be reconstructed exactly.

## Offline dataset visualization

Use the UMI-aware evaluator in this directory instead of
`examples/rtc/eval_dataset.py`. The generic evaluator compares unrelated random
samples and does not re-anchor an old UMI chunk into the current EE frame.

```bash
PYTHONPATH=src uv run python examples/umi_relative_ee/eval_rtc_dataset.py \
  --pretrained_path=/path/to/checkpoint/pretrained_model \
  --dataset_root=/path/to/validation_dataset \
  --repo_id=<dataset_repo_id> \
  --episode_indices 0 \
  --query_stride=5 \
  --inference_delay=4 \
  --execution_horizon=10 \
  --max_guidance_weight=10.0 \
  --device=cuda \
  --video_fps=6 \
  --project \
  --output_dir=outputs/debug/rtc_umi_dataset
```
`--project` uses the default SROI v2 hand-eye extrinsics and auto-discovers the
dataset's `camera_info_color.json`. Override `--extrinsics_config` or
`--camera_info_path` when using different calibration files.


For each sequential transition, the evaluator:

1. Predicts a preceding chunk and stores its postprocessed absolute EE targets.
2. Simulates executing `query_stride` actions.
3. Re-anchors the remaining targets into the current UMI chunk frame.
4. Runs no-RTC and RTC generation with identical sampled noise.
5. Saves camera-and-trajectory panels comparing the previous tail, no-RTC output,
   RTC output, and dataset ground truth.
6. Combines those panels into an H.264 episode MP4 and saves a JSON report with
   overlap, ground-truth, re-anchoring, roughness, and latency metrics.

RTC is intended to improve agreement with the still-executing previous chunk.
It is not guaranteed to reduce single-sample dataset ground-truth error.

## Verification coverage

`tests/policies/rtc/test_rtc_umi_relative_ee.py` covers both Pi0.5 and SmolVLA processor pipelines and verifies that:

1. Re-anchoring preserves each leftover action's absolute SE(3) target.
2. Rotation conversion uses the same row-based 6D convention as training.
3. Normalization matches the checkpoint processor statistics.
4. Moving the chunk-start pose changes the model-space prefix to the correct new frame.

## Relevant implementation files

- `examples/umi_relative_ee/eval_rtc_dataset.py`: sequential recorded-dataset metrics and visualization.
- `src/lerobot/rollout/inference/rtc.py`: RTC inference engine and UMI processor detection.
- `src/lerobot/policies/rtc/relative.py`: ordinary relative-action and UMI SE(3) re-anchoring helpers.
- `src/lerobot/processor/umi_relative_ee_processor.py`: `UmiRelativeActionsStep` and UMI SE(3) transforms.
- `src/lerobot/policies/pi05/modeling_pi05.py`: Pi0.5 RTC denoising hook.
- `src/lerobot/policies/smolvla/modeling_smolvla.py`: SmolVLA RTC denoising hook.
- `tests/policies/rtc/test_rtc_relative_actions.py`: ordinary relative-action RTC coverage.
- `tests/policies/rtc/test_rtc_umi_relative_ee.py`: Pi0.5 and SmolVLA UMI RTC re-anchoring coverage.
- `tests/processor/test_umi_relative_ee_processor.py`: general UMI processor coverage.

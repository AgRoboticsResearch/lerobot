# RTC with UMI Relative End-Effector Policies

This note describes whether Real-Time Chunking (RTC) can be used with Pi0.5 and SmolVLA checkpoints trained with the UMI relative end-effector pipeline in this directory.

## Current compatibility

| Policy and action representation | Current RTC status |
| --- | --- |
| Pi0, Pi0.5, or SmolVLA with absolute actions | Supported |
| Ordinary relative joint actions using `RelativeActionsProcessorStep` | Supported; the RTC rollout engine re-anchors the leftover prefix |
| Pi0.5 with `use_umi_relative_ee=true` | The model has an RTC hook, but guided RTC is not currently safe |
| SmolVLA with `use_umi_relative_ee=true` | The model has an RTC hook, but guided RTC is not currently safe |

**Do not enable guided RTC for a Pi0.5 or SmolVLA checkpoint trained with `use_umi_relative_ee=true` in the current implementation.** The limitation is in the inference-time coordinate conversion, not in the trained model or LoRA adapter.

## Why the current combination is incorrect

RTC guides the start of a newly generated chunk toward the unexecuted portion of the previous chunk. This requires both chunks to use the same action representation and reference frame.

UMI relative-EE preprocessing expresses every target in a chunk relative to the end-effector pose at the start of that chunk. For the same absolute target:

- A leftover action from the old chunk is represented as `T_old_base^-1 @ T_target`.
- The corresponding action in the new chunk must be represented as `T_current_base^-1 @ T_target`.

These relative values differ whenever the robot has moved between chunk starts. Comparing them directly does not preserve the same absolute target.

The RTC rollout engine currently detects `RelativeActionsProcessorStep` and re-anchors its leftover actions against the latest robot state. The UMI pipeline instead uses the separate `UmiRelativeActionsStep`, which RTC does not detect. As a result, the normalized UMI leftovers remain expressed in the old chunk frame when they are passed to RTC guidance. The guidance can therefore pull the new chunk toward the wrong absolute trajectory.

Pi0.5 and SmolVLA both expose native RTC denoising hooks, but those hooks operate on the prefix they receive and do not perform this UMI SE(3) frame conversion. The same limitation therefore applies to both policies.

## What can be used now

- Normal synchronous chunk inference remains valid for UMI-trained Pi0.5 and SmolVLA checkpoints.
- Asynchronous inference without cross-chunk guidance can still be implemented safely.
- Setting `execution_horizon=0` removes the incompatible RTC prefix guidance, but it also removes RTC transition smoothing. Treat this as asynchronous chunk generation, not full RTC.
- Absolute-action policies and ordinary relative-action policies using `RelativeActionsProcessorStep` can use the existing RTC path.

Changing the LoRA rank, training longer, or retraining the policy does not fix this issue. The correction belongs in the inference processor and queue path.

## Support required for UMI guided RTC

Existing UMI checkpoints should not require retraining once inference-side support is added. Before applying RTC guidance, the inference path must:

1. Read the leftover actions in absolute coordinates from the postprocessed action queue.
2. Re-express every leftover end-effector target relative to the latest UMI chunk-start pose using SE(3) composition.
3. Convert the relative rotations to the row-based 6D rotation representation used by the UMI training pipeline.
4. Normalize the re-anchored actions using the checkpoint's training statistics.
5. Pass this new model-space prefix to the Pi0.5 or SmolVLA RTC denoising hook.

The action queue must continue to keep both representations: model-space actions for denoising guidance and postprocessed absolute actions for robot execution and future re-anchoring.

## Testing requirement

The current tests cover RTC with ordinary relative actions and the UMI processor independently; they do not exercise UMI-relative-EE and RTC together. UMI RTC support should include an integration test that verifies:

1. Re-anchoring preserves each leftover action's absolute SE(3) target.
2. Rotation conversion uses the same row-based 6D convention as training.
3. Normalization matches the checkpoint processor statistics.
4. Both Pi0.5 and SmolVLA receive the correctly re-anchored prefix.

Until that implementation and integration test exist, use standard chunk inference for UMI relative-EE deployments.

## Relevant implementation files

- `src/lerobot/rollout/inference/rtc.py`: RTC inference engine and ordinary relative-action detection.
- `src/lerobot/policies/rtc/relative.py`: re-anchoring helper for `RelativeActionsProcessorStep`.
- `src/lerobot/processor/umi_relative_ee_processor.py`: `UmiRelativeActionsStep` and UMI SE(3) transforms.
- `src/lerobot/policies/pi05/modeling_pi05.py`: Pi0.5 RTC denoising hook.
- `src/lerobot/policies/smolvla/modeling_smolvla.py`: SmolVLA RTC denoising hook.
- `tests/policies/rtc/test_rtc_relative_actions.py`: ordinary relative-action RTC coverage.
- `tests/processor/test_umi_relative_ee_processor.py`: UMI processor coverage.

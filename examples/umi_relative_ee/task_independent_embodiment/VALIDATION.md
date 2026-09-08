# Implementation validation — 2026-09-07

The experiment is implemented and the full-run inputs are prepared. Full-budget
embodiment collection/training has **not** been launched.

## Completed

- Eight contract tests passed in the isolated simulator environment: delayed
  hindsight alignment, causal histories, interpolation/rot6d/frame conversion,
  future masking, provenance protection, paired episode bootstrap, deterministic
  simulator resets/command guards, source split isolation, and interrupted
  training matching uninterrupted training. Several contracts share a test.
- Complete CPU and RTX 4090 smoke matrices ran through native and 40 ms delayed
  dynamics, all four core methods, recollection, C1, the equal-data bootstrap
  control, frozen ACT inference, evaluation, and report generation.
- Final GPU smoke report:
  `/mnt/data1/projects/lerobot-embodiment-smoke/report/REPORT.md`.
  Its JSON reports `matrix_complete: true`. These are smoke results, not research
  evidence about controller quality.
- A full-size GPU training batch produced `[256, 5, 6]` outputs with finite
  gradients. Peak allocated memory was 1.31 GB; the first forward/backward pass
  took 0.23 seconds. This is a capacity check, not a full-run timing estimate.
- Full preflight measured maximum FK/site disagreement of approximately
  `2.49e-7 m` and `3.42e-5 degrees` across twenty deterministic configurations.
- Full configuration, all 1,459 source episodes, the 1,313/146 embodiment
  train/development split, and 500 frozen ACT queries covering all 100 validation
  episodes are ready at `/mnt/data1/projects/lerobot-embodiment-exp`.
- Ruff lint/format, shell syntax, and whitespace checks passed for the new code.

Run the full comparison matrix on the host with the existing prepared inputs:

```bash
bash examples/umi_relative_ee/task_independent_embodiment/run.sh all \
  --preset full --device cuda --resume
```

The persistent simulator environment is already installed at
`/mnt/data1/projects/lerobot-embodiment-env`. The launcher records console output
and exit codes under the artifact root's `logs/` directory.

## Existing test issue

The broader check covering the new tests, `test_eval_open_loop_dataset.py`, and
`test_umi_relative_ee_processor.py` returned 32 passed and one failure.
`test_summarize_reports_episode_balanced_primary_metric` in the existing
open-loop evaluator tests supplies a fixture without `rot_vel_deg_s` and other
physical-dynamics fields required by the existing `summarize` function. Neither
that fixture nor that summarizer was changed by this implementation.

The base LeRobot environment lacks SciPy; the new test module skips there.
Use the dedicated environment and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` as documented
in the README. The latter avoids an unrelated installed ROS pytest plugin whose
hook signature is incompatible with the experiment environment's pytest version.

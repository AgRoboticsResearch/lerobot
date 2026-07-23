# UMI legacy tool smoke test

Tested on 2026-07-23 from `fei-v5.0-umi-unified` with the py312 interpreter and
`PYTHONPATH` pointed at this worktree's `src`. The check imports each historical
CLI and invokes `--help`; it does not connect to hardware or execute motion.
All copied Python trees also pass `python -m compileall`.

## Result

- 24 command entrypoints pass their import/CLI smoke check (21 preserved tools plus
  all 3 unified training launchers).
- 7 preserved pre-5.0 investigation tools compile, but intentionally depend on
  the archived `RelativeEEDataset` or ACT temporal-wrapper implementation.
- 15 hardware/simulation tools compile, but their optional runtime dependency is
  not installed in the py312 environment.

The maintained training, conversion, deployment, visualization, and trajectory
entrypoints are in the passing group. The inactive dataset wrapper is not
registered into the 5.0 runtime: UMI training uses the raw absolute 7D dataset
plus the shared processor pipeline.

## Passing entrypoints

- `examples/so101/debug_act_so101_animation.py`
- `examples/so101/debug_act_so101_inference.py`
- `examples/so101/deploy_act_so101.py`
- `examples/so101_relative_ee/check_normalization_stats.py`
- `examples/so101_relative_ee/motion_frames/visualize_orb_traj.py`
- `examples/so101_relative_ee/visualize_camera_trajectory.py`
- `examples/so101_relative_ee/visualize_orbslam_projection.py`
- `examples/umi_relative_ee/deploy_relative_ee_processor_so101.py`
- `examples/umi_relative_ee/deploy_umi_relative_ee_piper.py`
- `examples/umi_relative_ee/gripper_control.py`
- `examples/umi_relative_ee/train_umi_relative_ee.py`
- `examples/umi_relative_ee/train_pi05_lora.py`
- `examples/umi_relative_ee/train_relative_ee_processor.py`
- `examples/umi_relative_ee/verify_pipeline_correctness.py`
- `examples/umi_relative_ee/visualize_predictions.py`
- `placo_sim/so101_deploy_real.py`
- `relative_ee_dataset/convert_ee_to_joint_dataset.py`
- `relative_ee_dataset/convert_joint_to_ee_dataset.py`
- `relative_ee_dataset/visualize_dataset_trajectories.py`
- `traj_execution_test/eetrajs/extract_gt_ee_trajectories.py`
- `traj_execution_test/piper_traj_test.py`
- `traj_execution_test/plot_compare_traj.py`
- `traj_execution_test/so101_chunked_traj_test.py`
- `traj_execution_test/so101_traj_test.py`

## Preserved archive-dependent entrypoints

These files remain source-compatible enough to compile, and their complete
supporting old implementation is preserved under `legacy/fei-relative-ee/`.
They are not made active because doing so would reintroduce a second dataset
transformation path alongside the unified processor contract.

| Entry point | Archived dependency |
|---|---|
| `examples/so101_relative_ee/debug_relative_ee_animation.py` | `lerobot.policies.act.temporal_wrapper` |
| `examples/so101_relative_ee/debug_relative_ee_dataloader.py` | `lerobot.datasets.relative_ee_dataset` |
| `examples/so101_relative_ee/debug_relative_ee_frame.py` | `lerobot.datasets.relative_ee_dataset` |
| `examples/so101_relative_ee/debug_relative_ee_inference.py` | `lerobot.policies.act.temporal_wrapper` |
| `examples/so101_relative_ee/debug_relative_ee_simulation.py` | `lerobot.policies.act.temporal_wrapper` |
| `examples/so101_relative_ee/deploy_relative_ee_so101_visualize.py` | `lerobot.datasets.relative_ee_dataset` |
| `examples/so101_relative_ee/visualize_camera_prediction.py` | `lerobot.datasets.relative_ee_dataset` |

## Optional dependency blockers

| Dependency | Entry points |
|---|---|
| `placo` | `examples/so101_relative_ee/deploy_relative_ee_so101.py`, `deploy_relative_ee_so101_static.py`, `replay_simulation.py`, `visualize_dataset_predictions.py`, `visualize_predictions.py`; `placo_sim/so101_deploy_sim.py` |
| `rerun` | `examples/so101_relative_ee/read_joints.py` |
| `pinocchio` | `placo_sim/so101_rel_motion_debug.py` |
| `zarr` | `relative_ee_dataset/convert_lerobot_to_umi.py` |
| `libero` | `traj_execution_test/libero_traj_test.py` |
| `mujoco` | `traj_execution_test/piper_mujoco_basic.py`, `piper_mujoco_traj_test.py`, `so101_mujoco_traj_test.py` |
| `piper_sdk` | `traj_execution_test/piper_print_state.py`, `piper_record_joint_limits.py` |

These are environment limitations, not missing migrated source. Hardware motion,
Placo IK, MuJoCo/Libero simulation, camera projection through the legacy tools,
and Piper SDK communication remain unexecuted on this machine. The maintained
unified visualizer itself imports successfully and supports calibration-based
projection through `--project`.

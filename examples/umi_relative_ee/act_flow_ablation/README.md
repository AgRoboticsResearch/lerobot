# ACT capacity and flow-objective ablation

This directory contains the controlled experiments answering two questions on
`sroiv2_strawberry_picking_lab_1459_occlusion`:

1. Does ACT improve with a larger visual backbone or transformer?
2. Is the observed VLA gap caused by flow matching itself, or by the VLM and its
   fine-tuning setup?

All runs use the same 1459-episode training set, 100-episode validation set,
UMI relative-EE rot6d representation, identity rotation normalization, action
chunk 30, batch size 8, and seed unless the run name says otherwise. Artifacts
default to `/media/zfei/Glowat512/projects/lerobot-arch-exp`; the source
filesystem has little free space and must not hold new checkpoint sweeps.

The complete decision log, literature analysis, experiment results, failures,
and lessons learned are consolidated in [`RESEARCH_REPORT.md`](RESEARCH_REPORT.md).

`act_r18_l1` and `act_r18_flow_*` are the decisive objective control. Both omit
the collapsed ACT VAE and share the same ResNet-18, observation transformer,
action decoder, data, and optimizer unless the variant names an LR change. The
flow version only adds noisy-action/time inputs and changes L1 regression to
rectified-flow velocity regression. `diffusion_r18` is a second, conventional
non-VLM control using the repository's ResNet + temporal U-Net Diffusion Policy.

Run on the host (not in a sandbox):

```bash
bash examples/umi_relative_ee/act_flow_ablation/run_one.sh act_r18_vae 30000 1000
bash examples/umi_relative_ee/act_flow_ablation/run_one.sh act_r34_vae 30000 1000
bash examples/umi_relative_ee/act_flow_ablation/run_one.sh act_r50_vae 30000 1000
bash examples/umi_relative_ee/act_flow_ablation/run_one.sh act_r18_l1 30000 1000
bash examples/umi_relative_ee/act_flow_ablation/run_one.sh act_r18_flow_u_lr1e5 30000 1000
bash examples/umi_relative_ee/act_flow_ablation/run_one.sh act_r18_flow_u_lr1e4 30000 1000
bash examples/umi_relative_ee/act_flow_ablation/run_one.sh act_r18_flow_beta_lr1e4 30000 1000
bash examples/umi_relative_ee/act_flow_ablation/run_one.sh diffusion_r18 30000 1000
```

Or run the fixed stage-one matrix sequentially, keeping only one process on the
GPU:

```bash
bash examples/umi_relative_ee/act_flow_ablation/run_stage1.sh 30000 1000
```

Evaluate a completed run on all 100 validation episodes with five fixed query
frames each:

```bash
bash examples/umi_relative_ee/act_flow_ablation/evaluate_one.sh \
  act_r18_vae_seed1000_30000steps 1000 5
```

The JSON report includes deterministic 95% bootstrap confidence intervals over
episodes. Repeat generative-policy evaluation with inference seeds 2000 and
3000; repeating deterministic ACT-L1/VAE inference is unnecessary.

Do not compare ACT L1, flow velocity MSE, and diffusion epsilon MSE numerically.
They have different scales. Compare checkpoints using decoded physical metrics
from `eval_open_loop_dataset.py`, fixing query frames and stochastic seeds.

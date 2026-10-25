# SmolVLA flow-matching padding A/B — final results (1M, complete)

**Date:** 2026-08-03. Both runs finished the full 1M steps.

Records the final validation-loss and endpoint-error numbers for the A/B
flow-matching-padding experiment, completing the picture sketched by the earlier
docs: formulation ([`padded_noise_strategy.md`](./padded_noise_strategy.md)),
early 100K ([`openpi_fullwidth_100k_results.md`](./openpi_fullwidth_100k_results.md)),
within-chunk jitter ([`within_chunk_jitter_analysis.md`](./within_chunk_jitter_analysis.md)),
and the transient offset ([`chunk_start_offset_transient.md`](./chunk_start_offset_transient.md)).

## Setup
- **Task:** SmolVLA UMI relative-EE, frozen VLM, on `sroiv2_strawberry_picking_lab_1302_occlusion`
  (train) + `sroiv2_strawberry_picking_lab_validation` (val, 100 eps).
- **Common config:** `policy.path=lerobot/smolvla_base`, chunk 30, batch 8, steps 1M,
  save_freq 100K, val_freq 10K, seed 1000, LR 1e-4 / warmup 1K / decay 30K → 2.5e-6.
  **Trainable params: 99.88 M of 450.05 M total (22%)** — action expert + state proj;
  VLM backbone (vision encoder + 16-layer SmolLM2) frozen.
- **Option A — `openpi_full_width`** (default): full-width 32-D Gaussian noise + loss over
  all 32 coords. `outputs/train/smolvla_openpi_fullwidth_1302_1M`.
- **Option B — `masked_subspace`**: noise/velocity/loss restricted to the real 10 dims;
  padded coords forced to 0 through train + inference. `outputs/train/smolvla_masked_subspace_1302_1M`.

## Validation loss (`flow_loss_real_dims` — the fair, full-val-set metric)
| run | best (step) | final @1M |
|-----|------|------|
| **A — openpi_full_width** | **0.0376** @50K | 0.0704 |
| **B — masked_subspace**   | **0.0400** @50K | 0.0785 |

A is marginally lower at both points but within ~6–10% → **essentially tied**. Both ~2× their
50K best by 1M (heavy overfit; best-val @50K is unsaved due to `save_freq=100K`, so the earliest
*saved* deployable ckpt is 100K).

## Endpoint error — open-loop prediction (val eps 0–2 only, 161 frames) @1M
| run | xyz (mm) | rot (°) | grip | …with `--zero_noise` (xyz / rot / grip) |
|-----|------|------|------|------|
| **A — openpi** val | 52.6 | 4.73 | 0.390 | 53.7 / **4.22** / 0.391 |
| **B — masked** val | 49.1 | 5.20 | 0.363 | 48.7 / **5.02** / 0.366 |

Train (eps 0–2, 215 frames) @1M for reference: A 15.8 mm / 1.60° (zn 16.5 / 1.44),
B 17.1 mm / 1.53° (zn 16.5 / 1.46).

Mixed at 1M — B better on val xyz, A better on val rot — so again **A ≈ B**.

## Conclusion
- **The masked-subspace formulation (B) gives no measurable benefit over openpi_full_width (A).**
  They tie on real-dim val loss, within-chunk jitter (see `within_chunk_jitter_analysis.md`),
  and endpoint accuracy. The original hypothesis (padded-noise coupling as the jitter root cause)
  is **not confirmed** — the within-chunk jitter is dominated by real-dimension flow-matching
  sampling noise, which both formulations share.
- **`--zero_noise` decoding** (all-zero flow latent) gives a small, consistent rotation-endpoint
  improvement (~0.3–0.5° val) with xyz unchanged — a diagnostic, not a deployment fix.
- **Both overfit by ~50K**; the 1M runs were ~20× past convergence. Deploy the earliest saved
  checkpoint (100K) or re-run with `save_freq=10000` to capture the ~50K best.
- The "fixed left move" once seen in B@300K viz was a **transient training artifact both A and B
  share** (~300K, resolves by 400–600K), not a masked-subspace bug — see
  `chunk_start_offset_transient.md`.

## Caveats
- Validation **loss** is the authoritative full-val-set number (computed every 10K by the trainer).
- The **endpoint errors** above are from open-loop viz on **val episodes 0–2 only** (161 frames),
  at the **final 1M (overfit)** checkpoint — a sample, not the full 100-episode val set, and not at
  the best-val region. A full-val endpoint eval (all 100 eps, at 100K or 1M) would be more rigorous.

## Checkpoints (on kiwi)
- A: `outputs/train/smolvla_openpi_fullwidth_1302_1M/checkpoints/{0100000…1000000,last}`
- B: `outputs/train/smolvla_masked_subspace_1302_1M/checkpoints/{0100000…1000000,last}`

---

# 中文翻译

> 以下为原文的完整中文翻译（AI 生成）。代码块、命令、参数、路径、URL、公式和数值保持原样；正文、标题、表格说明和列表文字译为中文。

# SmolVLA flow-matching padding A/B——最终结果（1M，已完成）

**日期：** 2026-08-03。两个运行均完成了全部 1M 步。

记录 A/B flow-matching-padding 实验的最终验证损失和端点误差数字，补全早期文档勾勒的图景：形式（[`padded_noise_strategy.md`](./padded_noise_strategy.md)）、早期 100K（[`openpi_fullwidth_100k_results.md`](./openpi_fullwidth_100k_results.md)）、块内抖动（[`within_chunk_jitter_analysis.md`](./within_chunk_jitter_analysis.md)）以及瞬态偏移（[`chunk_start_offset_transient.md`](./chunk_start_offset_transient.md)）。

## 设置
- **任务：** SmolVLA UMI relative-EE，冻结 VLM，在 `sroiv2_strawberry_picking_lab_1302_occlusion`（训练）+ `sroiv2_strawberry_picking_lab_validation`（验证，100 eps）上。
- **公共配置：** `policy.path=lerobot/smolvla_base`，chunk 30，batch 8，steps 1M，save_freq 100K，val_freq 10K，seed 1000，LR 1e-4 / warmup 1K / decay 30K → 2.5e-6。**可训练参数：450.05 M 中的 99.88 M（22%）**——动作专家 + 状态投影；VLM 骨干（视觉编码器 + 16 层 SmolLM2）冻结。
- **选项 A——`openpi_full_width`**（默认）：全宽 32-D 高斯噪声 + 覆盖全部 32 个坐标的损失。`outputs/train/smolvla_openpi_fullwidth_1302_1M`。
- **选项 B——`masked_subspace`**：噪声/速度/损失限制在真实的 10 维；填充坐标在训练 + 推理中强制为 0。`outputs/train/smolvla_masked_subspace_1302_1M`。

## 验证损失（`flow_loss_real_dims`——公平的全验证集指标）
| 运行 | 最佳（步数） | 1M 时最终 |
|-----|------|------|
| **A——openpi_full_width** | **0.0376** @50K | 0.0704 |
| **B——masked_subspace**   | **0.0400** @50K | 0.0785 |

A 在两个点上略低，但在 ~6–10% 以内 → **基本持平**。两者到 1M 时均约为其 50K 最佳的 2 倍（严重过拟合；最佳验证 @50K 因 `save_freq=100K` 未保存，因此最早的*已保存*可部署 ckpt 是 100K）。

## 端点误差——开环预测（仅 val eps 0–2，161 帧）@1M
| 运行 | xyz（mm） | 旋转（°） | grip | …带 `--zero_noise`（xyz / 旋转 / grip） |
|-----|------|------|------|------|
| **A——openpi** val | 52.6 | 4.73 | 0.390 | 53.7 / **4.22** / 0.391 |
| **B——masked** val | 49.1 | 5.20 | 0.363 | 48.7 / **5.02** / 0.366 |

训练（eps 0–2，215 帧）@1M 供参考：A 15.8 mm / 1.60°（zn 16.5 / 1.44），B 17.1 mm / 1.53°（zn 16.5 / 1.46）。

在 1M 时互有胜负——B 在 val xyz 上更好，A 在 val rot 上更好——因此再次是 **A ≈ B**。

## 结论
- **掩码子空间形式（B）相比 openpi_full_width（A）没有可测量的收益。** 它们在真实维度验证损失、块内抖动（见 `within_chunk_jitter_analysis.md`）和端点精度上打平。原始假设（填充噪声耦合是抖动根因）**未得到确认**——块内抖动主要由真实维度的 flow-matching 采样噪声主导，两种形式都共享该噪声。
- **`--zero_noise` 解码**（全零 flow latent）带来小幅、一致的旋转端点改善（val ~0.3–0.5°），xyz 不变——这是诊断手段，不是部署修复。
- **两者都在 ~50K 时过拟合**；1M 运行已超过收敛约 20 倍。部署最早的已保存检查点（100K），或用 `save_freq=10000` 重新运行以捕获 ~50K 最佳。
- B@300K 可视化中曾见到的“固定左移”是 **A 和 B 共享的瞬态训练伪影**（~300K，400–600K 消退），不是掩码子空间缺陷——见 `chunk_start_offset_transient.md`。

## 注意事项
- 验证**损失**是权威的全验证集数字（由训练器每 10K 计算）。
- 上述**端点误差**来自**仅 val episodes 0–2**（161 帧）的开环可视化，处于**最终 1M（过拟合）**检查点——是样本，不是完整 100 episode 验证集，也不在最佳验证区域。全验证集端点评估（全部 100 eps，在 100K 或 1M）会更严谨。

## 检查点（在 kiwi 上）
- A: `outputs/train/smolvla_openpi_fullwidth_1302_1M/checkpoints/{0100000…1000000,last}`
- B: `outputs/train/smolvla_masked_subspace_1302_1M/checkpoints/{0100000…1000000,last}`

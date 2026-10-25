# lerobot fei-v5.0-umi-unified (0.5.2) 代码阅读总结

> 2026-08-01，用户睡觉时 AI 自主阅读 + 总结。供下次会话参考。

## 一、环境现状

| 项 | 值 |
|---|---|
| 分支 | `fei-v5.0-umi-unified`（lerobot 0.5.2） |
| 环境 | conda `lerobot`（python 3.12 + torch 2.10+cu128 + fork 0.5.2 editable） |
| GPU | RTX 5060 Laptop（sm_120，cu128 必须） |
| sroi 数据处理 | LeRobotDataset + Rotation（0.5.2 兼容 0.5.1）✓ |
| 旧环境 lerobot_fork（0.4.3 python 3.10）| 已删 |
| .local lerobot 0.5.1 | 已删（fork 0.5.2 取代） |
| requirements-ubuntu.txt pip 解析 | 有冲突（dependency resolver too deep），但核心依赖（torch/lerobot/transformers/datasets）已在 |

## 二、训练数据

| 数据集 | 路径 | 规模 |
|---|---|---|
| train | `~/data/lerobot/strawberry_picking_full` | 1135 ep / 102448 frames |
| val | `~/data/lerobot/strawberry_picking_val` | 100 ep / 9274 frames（独立另一天采，和 scaling 报告一致） |

数据 schema: observation.images.camera(video) + observation.ee[7] + action[7]（绝对 EE）。

## 三、0.5.2 核心变化（vs 旧 fei 0.4.3）

### 统一入口
所有策略（ACT/SmolVLA/π0.5）用同一标志 `--policy.use_umi_relative_ee=true`：
```bash
lerobot-train --policy.use_umi_relative_ee=true --policy.type=act ...
```
旧入口（train_relative_ee.py, train_relative_ee_processor.py）变 thin wrapper。

### 数据契约（不变）
- 数据集存绝对 7D action
- processor 在线算相对：state=20D（两帧 rot6d），action=10D 相对
- chunk 内所有动作相对 chunk 起点 base pose

### 新增 π0.5 + LoRA
train_pi05_lora.py/sh。需要 HF 授权（PaliGemma gated）。

### 新增异步推理
async_umi_relative_ee_policy_server.py + async_umi_relative_ee_piper_client.py

### RelativeEEDataset 不再存在
0.5.2 用 processor pipeline（umi_relative_ee_stats.py + normalize_processor）替代了旧的 RelativeEEDataset 类。数学等价，实现路径不同。

## 四、rotation_normalization.md 核心结论

**问题**：我们的 pipeline 对 rot6d 的 6 个维度做了 per-dim MIN_MAX/QUANTILES 归一化，但 UMI 原版对 rot6d 用 **identity（不归一化）**。

**为什么有害**：相对动作的 rot6d 接近 identity `[1,0,0,0,1,0]`：
- 对角维度（rot6d[0]≈1, rot6d[4]≈1）方差极小 → per-dim 归一化把它们**拉伸到 [-1,1]**，放大了近常数信号
- 非对角维度（编码实际旋转方向）方差大 → 归一化后权重小
- **结果**：loss 被近常数对角维度主导，真正有用的旋转方向欠拟合 → **旋转抖动（jumpiness）**

**这也解释了 ACT VAE 坍塌**：VAE 的 KL loss 在归一化扭曲的 rot6d 上计算，后验坍塌到 prior（KLD→0），VAE 退化成纯 transformer。

**修复方案（可逆实验）**：在 stats 里强制 rot6d dims 为 identity（q01=-1, q99=1），不做 per-dim 缩放。A/B 测试对比。

**和我们之前 state identity 讨论的关系**：state identity 占位（observation.state = identity）是设计选择（相对动作不需要绝对位姿输入）。rotation normalization 问题是 rot6d **归一化方式**的问题——两者不同层面，但都指向"UMI 风格设计在实现中的偏差"。

## 五、examples/umi_relative_ee/ 文件索引

| 文件 | 用途 |
|---|---|
| README.md | **总览**（安装/训练/验证/部署契约） |
| train_relative_ee_processor.py | ACT/SmolVLA 训练入口（wrapper） |
| train_pi05_lora.py / .sh | π0.5 LoRA 训练 |
| train_umi_relative_ee.py | 统一训练 wrapper |
| deploy_umi_relative_ee_piper.py | Piper 同步部署 |
| async_umi_relative_ee_policy_server.py | 异步推理服务器 |
| async_umi_relative_ee_piper_client.py | 异步 Piper 客户端 |
| deploy_relative_ee_processor_so101.py | SO101 部署 |
| eval_open_loop_dataset.py | 策略中性评估（ACT/SmolVLA/π0.5） |
| eval_rtc_dataset.py | RTC 评估（SmolVLA/π0.5） |
| visualize_predictions.py | 预测可视化 |
| gripper_control.py | 夹爪控制工具 |
| camera_gripper_extrinsics_sroi_v2_d405.json | D405 手眼标定 |
| umi_style_ee_processor_pipeline.md | processor 设计文档 |
| rotation_normalization.md | **旋转归一化分析**（核心问题） |
| rot6d_identity_norm_experiment.md | identity 归一化 A/B 实验 |
| smolvla_relative_ee_training.md | SmolVLA 训练指南 |
| ASYNC_INFERENCE.md | 异步推理文档 |
| prediction_visualization.md | 可视化文档 |
| rtc.md | RTC 部署文档 |
| scaling/ | scaling 实验报告（100/500/1012 ep） |
| within_chunk_jitter_analysis.md | chunk 内抖动分析 |
| pi0.5_finetunning.md | π0.5 微调指南 |

## 六、训练命令（0.5.2 统一入口，已验证 ✅ 2026-08-01）

dry run（ACT + UMI relative EE + 20 步 + 3 个 val ep）已跑通，exit 0。**必须同时设 `chunk_size=30` + `n_action_steps=30`**（ACT 默认 100/100，只设 chunk_size 会触发 `n_action_steps > chunk_size` 报错）；**本地训练必须 `--policy.push_to_hub=false`**（否则要求 policy.repo_id 才能推 HF）。

```bash
cd ~/lerobot
conda run -n lerobot --no-capture-output python examples/umi_relative_ee/train_umi_relative_ee.py \
  --policy.type=act \
  --policy.use_umi_relative_ee=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=sroi/strawberry_picking_full \
  --dataset.root=/home/ss/data/lerobot/strawberry_picking_full \
  --dataset.use_imagenet_stats=true \
  --validation_dataset.repo_id=sroi/strawberry_picking_val \
  --validation_dataset.root=/home/ss/data/lerobot/strawberry_picking_val \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --batch_size=8 \
  --num_workers=4 \
  --steps=180000 \
  --val_freq=10000 --eval_freq=0 --log_freq=500 \
  --save_checkpoint=true \
  --wandb.enable=false \
  --output_dir=outputs/train/act_umi_v2
```

- `disable_temporal_wrapper`（默认 True）、`use_joint_obs`（默认 False）都不用手动设；UMI 模式在 `ACTConfig.__post_init__` 自动开 `derive_state_from_action/use_relative_actions/pose_dim=6/use_rot6d` + MIN_MAX（STATE/ACTION）。
- rotation normalization A/B：加 `--policy.umi_rot6d_identity_norm=true`（强制 rot6d 用 identity stats，见 `rotation_normalization.md`）。
- 真实训练用 `examples/umi_relative_ee/train_umi_relative_ee.py`（thin wrapper）等价于标准 `lerobot-train` + `--policy.use_umi_relative_ee=true`，不再是旧 monkey-patch。

### Dry run 实测数据（2026-08-01，batch=4，20 步）
- UMI processor 正常：`raw action 7D -> model action 10D, derived state 20D` ✓
- train loss 68.9 → 30.4（在学）；val（3 ep）step10 loss=29.1/kld=2.88，step20 loss=16.1/kld=1.58（20 步内 KL 活跃，未坍塌；是否最终坍塌见 scaling 报告）
- 速度：稳态 ~5–6 step/s（updt 0.08s，data 0.003s）。**外推：180k 步 ≈ 9–10 小时**（scaling 报告的 1012-ep 高效点，远好过 2.5M 全量）。batch=8 会略慢但 VRAM 够（RTX 5060 8GB 建议先用 4 试，不 OOM 再加）。

## 七、待办

- [x] **训练 dry run 验证（0.5.2 统一入口能跑，ACT 全链路通）** ✅ 2026-08-01
- [ ] requirements-ubuntu.txt pip 解析冲突（核心依赖已够训练，缺包用 uv sync 补）
- [ ] rotation_normalization A/B 实验（加 `--policy.umi_rot6d_identity_norm=true`，**该 flag 已验证可跑** ✅；20 步 dry run 显示 identity 下 KLD 低 7 倍但 l1 不变，20 步太小不能下结论，需完整训练对比 jumpiness）
- [ ] π0.5 依赖 + HF 授权（PaliGemma gated，当前机器 403；要训 π0.5 再处理）
- [ ] 首次真实训练（稳定 10 小时窗口，180k 步，batch 从 4 起步）

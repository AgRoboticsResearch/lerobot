# Async inference for UMI relative-EE on Piper

This guide is the asynchronous equivalent of:

```bash
python examples/umi_relative_ee/deploy_umi_relative_ee_piper.py \
  --pretrained_path outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30_sroi_v2_masked_1012train_100val/checkpoints/2500000/pretrained_model/ \
  --cameras="{camera: {type: intelrealsense, fps: 30, width: 640, height: 480}}" \
  --n_action_steps=20
```

The async version uses two processes:

```text
Piper + cameras + FK/IK + safety                 Policy + GPU
async_umi_relative_ee_piper_client.py   gRPC    async_umi_relative_ee_policy_server.py
        execute queued actions             <---       return absolute 7D chunks
        send two-pose state + images        --->       preprocess + infer + postprocess
```

The robot continues executing its current action queue while the server predicts
the next chunk.

## 1. Install the async dependency

Use the same environment in which the synchronous Piper command already works:

```bash
uv sync --locked --extra async
```

This adds gRPC and the optional queue-size plot. The Piper SDK, RealSense
dependency, CAN setup, calibration, URDF, and external gripper setup are the
same as for `deploy_umi_relative_ee_piper.py`.

## 2. Run on one machine

Open two terminals in the repository root.

### Terminal 1: start the policy server

```bash
uv run python examples/umi_relative_ee/async_umi_relative_ee_policy_server.py \
  --host=127.0.0.1 \
  --port=8080 \
  --fps=30
```

The server starts empty. It loads the checkpoint after the client connects and
sends the policy configuration.

### Terminal 2: start the Piper client

```bash
uv run python examples/umi_relative_ee/async_umi_relative_ee_piper_client.py \
  --server_address=127.0.0.1:8080 \
  --pretrained_path=outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30_sroi_v2_masked_1012train_100val/checkpoints/2500000/pretrained_model/ \
  --policy_type=act \
  --policy_device=cuda \
  --cameras="{camera: {type: intelrealsense, fps: 30, width: 640, height: 480}}" \
  --n_action_steps=20 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=latest_only
```

`--n_action_steps` is an alias for `--actions_per_chunk`. The command above
therefore requests 20 actions from the checkpoint's 30-action prediction
horizon, matching the synchronous command.

The client connects to the server and waits while the server loads the model.
It then connects the Piper, gripper, and camera, moves to the configured start
pose, and enters `PAUSED`.

Press `s` only after the work area is clear.

## 3. Keyboard controls

| Key | Behavior |
| --- | --- |
| `s` | Discard old actions, request a fresh chunk, and engage async control |
| Space | Pause immediately and invalidate queued/in-flight actions |
| `.` | Request and execute exactly one fresh chunk, then pause |
| `q` | Move to the start pose and remain paused |
| `r` | Move to the safe pose and remain paused |
| `h` | Print the key map |
| Esc | Stop, return to the safe pose, and disconnect |

The client rejects a delayed server response if it was generated before a
pause, start-pose move, or safe-pose move. This prevents a stale chunk based on
the old robot pose from being executed after a direct move.

## 4. Run the server on another machine

Both machines need this repository version and the async dependencies. Only the
client machine needs the Piper, CAN, gripper, and camera hardware.

On the GPU/server machine:

```bash
uv run python examples/umi_relative_ee/async_umi_relative_ee_policy_server.py \
  --host=0.0.0.0 \
  --port=8080 \
  --fps=30
```

On the robot/client machine, replace the address with the server's LAN IP:

```bash
uv run python examples/umi_relative_ee/async_umi_relative_ee_piper_client.py \
  --server_address=192.168.1.50:8080 \
  --pretrained_path=/path/on/server/to/pretrained_model \
  --policy_type=act \
  --policy_device=cuda \
  --cameras="{camera: {type: intelrealsense, fps: 30, width: 640, height: 480}}" \
  --n_action_steps=20 \
  --chunk_size_threshold=0.5
```

Important:

- `--pretrained_path` is opened by the **server**, not the robot client. It
  must be a path that exists on the server, or a Hugging Face model ID.
- The camera names must match the checkpoint inputs. A checkpoint trained with
  `observation.images.camera` needs a client camera named `camera`.
- Allow TCP port 8080 through the server firewall.
- This uses insecure gRPC. Use it only on a trusted LAN or through a VPN/SSH
  tunnel; do not expose the port directly to the internet.

## 5. Why this needs UMI-specific scripts

The standard LeRobot async client assumes a registered `Robot` whose state is
assembled from motor features. This deployment instead computes a 7D absolute
EE pose with Piper FK:

```text
[x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper]
```

The UMI checkpoint also requires two adjacent control-loop poses. The custom
client sends `[previous, current]` on every inference request, preserving the
30 Hz state history even when inference requests are less frequent.

On the server, the checkpointed preprocessor converts that pair to the policy's
20D relative state. The server then postprocesses the entire predicted chunk in
one call, so all 7D absolute targets use the same chunk-start reference. The
client converts each received target through EE bounds checks and IK before
writing Piper joint commands.

## 6. Tune the queue

Start with:

```text
actions_per_chunk=20
chunk_size_threshold=0.5
aggregate_fn_name=latest_only
fps=30
```

- `actions_per_chunk` is how many actions the server returns. It must not exceed
  the checkpoint's maximum prediction horizon. Larger chunks provide more
  latency margin but use older predictions for longer.
- `chunk_size_threshold=0.5` requests a new prediction when half of the current
  chunk remains. Increase it toward `0.7` if the queue often empties. This
  increases inference frequency and overlap.
- Lower `fps` if the server cannot produce a new chunk before the client queue
  reaches zero.
- `latest_only` replaces overlapping old targets with the newest prediction.
  It is the default for this client because component-wise averaging of
  absolute axis-angle poses can behave poorly near representation
  discontinuities.

To inspect queue behavior, add:

```bash
--debug_visualize_queue_size
```

The queue plot appears after graceful shutdown. A healthy trace is refilled
before it repeatedly reaches zero.

## 7. Common problems

### Client cannot connect

Confirm the server is already running, the address and port match, and the
firewall allows the connection:

```bash
ss -ltn | grep 8080
```

### Server reports that the checkpoint does not exist

Use a path on the server machine. A path that exists only on the robot machine
will not work.

### Server reports a missing policy camera

Make the key in `--cameras` match the checkpoint's image feature name. For the
example checkpoint this is normally `camera`.

### Queue repeatedly reaches zero

Try, in order:

1. Increase `--chunk_size_threshold` from `0.5` to `0.6` or `0.7`.
2. Increase `--n_action_steps`, up to the policy horizon.
3. Lower `--fps`.
4. Reduce network latency or use a faster inference device.

### A chunk is logged with `accepted=False`

This is expected after pausing or moving directly to the start/safe pose. The
client is rejecting a response generated from a stale pre-move observation.

### The arm does not move after startup

The client deliberately starts paused. Press `s` to engage, or use `.` for the
safer first test that executes one fresh chunk and pauses again.

## 8. Recommended first test

1. Clear the robot workspace and keep an emergency stop available.
2. Start the server.
3. Start the client and wait for model and hardware initialization.
4. Press `.` to execute one fresh 20-action chunk.
5. Verify the direction, scale, gripper behavior, IK output, and queue logs.
6. Press `s` only after the one-chunk test behaves correctly.

---

# 中文翻译

> 以下为原文的完整中文翻译（AI 生成）。代码块、命令、参数、路径、URL、公式和数值保持原样；正文、标题、表格说明和列表文字译为中文。

# 在 Piper 上进行 UMI relative-EE 的异步推理

本指南是以下同步命令的异步等价版本：

```bash
python examples/umi_relative_ee/deploy_umi_relative_ee_piper.py \
  --pretrained_path outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30_sroi_v2_masked_1012train_100val/checkpoints/2500000/pretrained_model/ \
  --cameras="{camera: {type: intelrealsense, fps: 30, width: 640, height: 480}}" \
  --n_action_steps=20
```

异步版本使用两个进程：

```text
Piper + cameras + FK/IK + safety                 Policy + GPU
async_umi_relative_ee_piper_client.py   gRPC    async_umi_relative_ee_policy_server.py
        execute queued actions             <---       return absolute 7D chunks
        send two-pose state + images        --->       preprocess + infer + postprocess
```

机器人在服务器预测下一个 chunk 的同时继续执行当前的动作队列。

## 1. 安装异步依赖

使用同步 Piper 命令已经可以正常工作的同一环境：

```bash
uv sync --locked --extra async
```

这会添加 gRPC 和可选的队列大小图。Piper SDK、RealSense 依赖、CAN 设置、标定、URDF 以及外部 gripper 设置都与 `deploy_umi_relative_ee_piper.py` 相同。

## 2. 在单机上运行

在仓库根目录打开两个终端。

### 终端 1：启动策略服务器

```bash
uv run python examples/umi_relative_ee/async_umi_relative_ee_policy_server.py \
  --host=127.0.0.1 \
  --port=8080 \
  --fps=30
```

服务器启动时为空。它在客户端连接并发送策略配置之后加载 checkpoint。

### 终端 2：启动 Piper 客户端

```bash
uv run python examples/umi_relative_ee/async_umi_relative_ee_piper_client.py \
  --server_address=127.0.0.1:8080 \
  --pretrained_path=outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30_sroi_v2_masked_1012train_100val/checkpoints/2500000/pretrained_model/ \
  --policy_type=act \
  --policy_device=cuda \
  --cameras="{camera: {type: intelrealsense, fps: 30, width: 640, height: 480}}" \
  --n_action_steps=20 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=latest_only
```

`--n_action_steps` 是 `--actions_per_chunk` 的别名。因此上面的命令从 checkpoint 的 30 个动作预测视野中请求 20 个动作，与同步命令一致。

客户端连接到服务器，并在服务器加载模型期间等待。随后它连接 Piper、gripper 和相机，移动到配置的起始位姿，并进入 `PAUSED` 状态。

只有在工作区域清空后，才按 `s`。

## 3. 键盘控制

| 按键 | 行为 |
| --- | --- |
| `s` | 丢弃旧动作，请求一个新的 chunk，并启用异步控制 |
| Space | 立即暂停并使排队中/正在执行的动作失效 |
| `.` | 请求并精确执行一个新鲜的 chunk，然后暂停 |
| `q` | 移动到起始位姿并保持暂停 |
| `r` | 移动到安全位姿并保持暂停 |
| `h` | 打印键位映射 |
| Esc | 停止，返回安全位姿，并断开连接 |

如果服务器的延迟响应是在暂停、移动到起始位姿或移动到安全位姿之前生成的，客户端会拒绝该响应。这可防止基于旧机器人位姿的过期 chunk 在直接移动之后被执行。

## 4. 在另一台机器上运行服务器

两台机器都需要此仓库版本和异步依赖。只有客户端机器需要 Piper、CAN、gripper 和相机硬件。

在 GPU/服务器机器上：

```bash
uv run python examples/umi_relative_ee/async_umi_relative_ee_policy_server.py \
  --host=0.0.0.0 \
  --port=8080 \
  --fps=30
```

在机器人/客户端机器上，将地址替换为服务器的局域网 IP：

```bash
uv run python examples/umi_relative_ee/async_umi_relative_ee_piper_client.py \
  --server_address=192.168.1.50:8080 \
  --pretrained_path=/path/on/server/to/pretrained_model \
  --policy_type=act \
  --policy_device=cuda \
  --cameras="{camera: {type: intelrealsense, fps: 30, width: 640, height: 480}}" \
  --n_action_steps=20 \
  --chunk_size_threshold=0.5
```

重要说明：

- `--pretrained_path` 由**服务器**打开，而不是机器人客户端。它必须是服务器上存在的路径，或一个 Hugging Face 模型 ID。
- 相机名称必须与 checkpoint 输入匹配。使用 `observation.images.camera` 训练的 checkpoint 需要名为 `camera` 的客户端相机。
- 在服务器防火墙上放行 TCP 端口 8080。
- 这使用的是不安全的 gRPC。只应在可信局域网中或通过 VPN/SSH 隧道使用；不要将该端口直接暴露到互联网。

## 5. 为什么这需要 UMI 专用脚本

标准 LeRobot 异步客户端假设存在一个已注册的 `Robot`，其状态由电机特征组装而成。而本部署改为使用 Piper FK 计算 7D 绝对 EE 位姿：

```text
[x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper]
```

UMI checkpoint 还需要两个相邻控制循环的位姿。自定义客户端在每次推理请求时发送 `[previous, current]`，即使推理请求频率较低，也能保留 30 Hz 的状态历史。

在服务器上，checkpoint 中的预处理器将该对位姿转换为策略的 20D 相对状态。然后服务器在一次调用中对整个预测 chunk 进行后处理，因此全部 7D 绝对目标使用相同的 chunk 起始参考。客户端在写入 Piper 关节命令之前，会通过 EE 边界检查和 IK 转换每个接收到的目标。

## 6. 调整队列

从以下配置开始：

```text
actions_per_chunk=20
chunk_size_threshold=0.5
aggregate_fn_name=latest_only
fps=30
```

- `actions_per_chunk` 是服务器返回的动作数量。它不能超过 checkpoint 的最大预测视野。更大的 chunk 提供更多延迟余量，但会更长时间地使用较旧的预测。
- `chunk_size_threshold=0.5` 表示当当前 chunk 还剩一半时请求新的预测。如果队列经常耗尽，可将其向 `0.7` 提高。这会增加推理频率和重叠。
- 如果服务器无法在客户端队列降到零之前生成新的 chunk，请降低 `fps`。
- `latest_only` 用最新的预测替换重叠的旧目标。它是此客户端的默认值，因为绝对轴角位姿的逐分量平均在表示不连续附近可能表现不佳。

要检查队列行为，请添加：

```bash
--debug_visualize_queue_size
```

队列图会在优雅关闭后出现。健康的轨迹会在反复降到零之前得到补充。

## 7. 常见问题

### 客户端无法连接

确认服务器已在运行、地址和端口匹配，并且防火墙允许连接：

```bash
ss -ltn | grep 8080
```

### 服务器报告 checkpoint 不存在

请使用服务器机器上的路径。仅存在于机器人机器上的路径无法工作。

### 服务器报告缺少策略相机

使 `--cameras` 中的键与 checkpoint 的图像特征名称匹配。对于示例 checkpoint，这通常是 `camera`。

### 队列反复降到零

按顺序尝试：

1. 将 `--chunk_size_threshold` 从 `0.5` 提高到 `0.6` 或 `0.7`。
2. 增大 `--n_action_steps`，直至策略视野上限。
3. 降低 `--fps`。
4. 降低网络延迟或使用更快的推理设备。

### 某个 chunk 被记录为 `accepted=False`

在暂停或直接移动到起始/安全位姿之后，这是预期行为。客户端正在拒绝基于移动前的过期观测生成的响应。

### 启动后机械臂不动

客户端有意以暂停状态启动。按 `s` 启用控制，或使用 `.` 进行更安全的首次测试：执行一个新鲜的 chunk 并再次暂停。

## 8. 建议的首次测试

1. 清理机器人工作空间，并确保急停可用。
2. 启动服务器。
3. 启动客户端，等待模型和硬件初始化。
4. 按 `.` 执行一个新鲜的 20 动作 chunk。
5. 验证方向、尺度、gripper 行为、IK 输出和队列日志。
6. 只有在单 chunk 测试表现正确后，才按 `s`。

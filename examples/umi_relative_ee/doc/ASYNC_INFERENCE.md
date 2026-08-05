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

# G1 SONIC 独立采集操作包

这个目录把当前已经验证过的 G1 + PICO + Inspire 右手 + HTTP 相机采集流程封装成一个命令行工具。日常采集不需要再让 GPT 拼接命令。

脚本仍然保留三个人工安全动作：物理急停、PICO 的 `Send`、控制器组合键。脚本不会自动绕过这些门槛。

熟悉完整流程后，可直接逐项照着 `OPERATOR_CHECKLIST.md` 执行。

## 最短执行清单

第一次使用仍请完整阅读后文。以后每次采集只需按下面顺序逐条执行；上一条出现 `[FAIL]` 就停止，不执行下一条。

```bash
cd /home/data-collection/GR00T-WholeBodyControl/tools/0902collect

./collectctl.sh init --date 09_06 --action pick --object box --run 1 \
  --task 'pick up the box' --hand-mode right-only \
  --camera-url 'http://222.29.157.156:8765/stream.mjpg'

./collectctl.sh config
./collectctl.sh pc-service
# 此时在 PICO 中校正并点击 Send
./collectctl.sh launch
```

`launch` 会自动执行完整 `preflight`，其中已经包含连续 12 秒的 PICO 检查；正常流程不需要提前重复执行。`pico-check` 和 `preflight` 保留给单项诊断。

进入 tmux 后按顺序完成人工动作：pane 0 按 Enter，等待 `Init Done`；四键 `A+B+X+Y` 启动并松开；`A+X` 进入 POSE 并松开。第一次录制前先用 `Ctrl+b d` 脱离 tmux，执行实时健康门禁：

```bash
./collectctl.sh health
```

只有最后出现 `[PASS] collection is healthy; recording may start or continue` 才能重新 `attach`，再用左 Grip + 右 A 开始/停止录制。

每个 episode 保存后可脱离 tmux 并检查：

```bash
./collectctl.sh status
./collectctl.sh verify-latest
./collectctl.sh attach
```

结束策略并正常保存后执行：

```bash
./collectctl.sh stop
```

## 0. 物理连接和上电

1. 清空机器人和灵巧手周围的人、箱子、线缆和工具，急停拿在手中。
2. ThinkPad 的 USB 网卡接机器人交换机。
3. G1、左手控制器、右手控制器接同一交换机；右手单手采集时左手可以不接。
4. 交换机、G1 和所需手控制器上电，等待 G1 启动完成。
5. ThinkPad 和 PICO 接同一个 Wi-Fi。
6. HTTP 相机服务器保持运行；当前地址为 `http://222.29.157.156:8765/stream.mjpg`。

电脑只需打开一个终端运行本目录的命令；`launch` 会自动建立 tmux 的四个 pane 和两个附加窗口，不需要手工再开四个终端。HTTP 相机服务器位于另一台设备时，仍需在该设备上单独保持相机服务运行。

## 1. 创建本次会话配置

进入操作包：

```bash
cd /home/data-collection/GR00T-WholeBodyControl/tools/0902collect
```

推荐用命名参数自动生成数据集名：

```bash
./collectctl.sh init \
  --date 09_06 \
  --action pick \
  --object box \
  --run 1 \
  --task 'pick up the box' \
  --hand-mode right-only \
  --camera-url 'http://222.29.157.156:8765/stream.mjpg'
```

它生成 `session.env`，数据集名称是：

```text
09_06_pick_box_1
│     │    │   └─ 同一天同一任务的第几次独立采集
│     │    └──── 物体或场景短名
│     └───────── 动作短名
└─────────────── 月_日
```

`--date` 可以省略，脚本会使用电脑当前日期；`--action`、`--object` 只用英文、数字或短横线，`--run` 必须是从 1 开始的正整数。相同日期、动作和物体的下一批数据只增加 `--run`，不要复用旧名字。

命名和会话参数：

| `init` 参数 | 示例 | 含义 |
|---|---|---|
| `--date` | `09_06` | 月和日；省略时取电脑当天日期 |
| `--action` | `pick` | 动作短名，用于生成目录名 |
| `--object` | `box` | 物体/场景短名，用于生成目录名 |
| `--run` | `1` | 同名任务的独立批次编号，不是 episode 编号 |
| `--task` | `pick up the box` | 写入数据集的语言任务，不能为空 |
| `--hand-mode` | `right-only` | `right-only`、`left-only` 或 `both` |
| `--camera-url` | `222.29.157.156:8765/stream.mjpg` | ego-view HTTP MJPEG 数据源 |
| `--dataset` | `09_06_pick_box_1` | 不按三段式生成时，直接指定完整目录名 |
| `--resume` | 无参数值 | 续采已有目录；不写时默认为新建 |

也可以直接指定名字：

```bash
./collectctl.sh init \
  --dataset 09_06_pick_box_1 \
  --task 'pick up the box' \
  --hand-mode right-only \
  --camera-url 'http://222.29.157.156:8765/stream.mjpg'
```

续采已有数据集必须显式加 `--resume`：

```bash
./collectctl.sh init \
  --dataset 09_06_pick_box_1 \
  --task 'pick up the box' \
  --hand-mode right-only \
  --camera-url 'http://222.29.157.156:8765/stream.mjpg' \
  --resume
```

脚本不会自动覆盖已有数据集。新建模式发现同名目录会失败；续采模式发现目录不存在、文件不完整，或任务、手模式、相机、帧率、模型配置与原数据集不一致时也会失败。

检查最终配置：

```bash
./collectctl.sh config
```

应看到数据集、任务文本、`right-only`、相机 URL、G1/手 IP、模型、频率，以及当前手姿态文件和其中的 `open/grasp/thumb_rotate` 摘要。`PICO PC Service IP` 是当前 ThinkPad Wi-Fi 地址，DHCP 变化后以这里显示的值为准。

当前采集专用的灵巧手开合配置是：

```text
/home/data-collection/GR00T-WholeBodyControl/tools/0902collect/inspire_hand_pose.json
```

六个数组元素依次是 `little`、`ring`、`middle`、`index`、`thumb_bend`、`thumb_rotate`。本项目使用的角度寄存器约定为 `1.0 = 张开/向外`、`0.0 = 闭合/向内`：

- `open`：PICO 请求张开时的六通道目标。
- `grasp`：PICO 请求闭合时的六通道目标。
- `thumb_rotate_default`：非 `null` 时固定覆盖第六通道；当前为 `1.0`，因此启动时和后续开合过程中都保持拇指横向最大。设为 `null` 时，第六通道分别跟随 `open` 和 `grasp` 数组。

只在采集栈完全停止时编辑该文件；修改后先运行 `./collectctl.sh config`，确认摘要正确，再运行 `launch`。手桥和导出器只在启动时读取配置，运行中修改不会热更新。任何数值必须在 `[0.0, 1.0]` 内，且两个数组都必须恰好有六项。执行 `launch` 前要清空右手周围并把急停放在手边。

## 2. 启动 PC Service，PICO 校正并 Send

```bash
./collectctl.sh pc-service
```

应看到：

```text
[PASS] RoboticsService ...
[PASS] set PICO PC Service to <当前 ThinkPad Wi-Fi IP>, recalibrate, then press Send
```

在 PICO/XRoboToolkit 内：

1. 把 PC Service 地址设置为脚本显示的 IP。
2. 选择身体跟踪设备。
3. 重新校正。
4. 点击一次 `Send`。

需要单独诊断 PICO 时运行：

```bash
./collectctl.sh pico-check
```

必须看到：

```text
body=True trackers=... duration=12s ... advanced=True max_timestamp_stale_ms=...
[PASS] PICO body data is live
```

其中 `body=True`、`advanced=True` 且 `max_timestamp_stale_ms` 不超过 800 ms。`trackers=0` 在身体骨架正常时不是故障。该检查会连续观察 12 秒，不是只取两个瞬时样本。800 ms 是启动检查门槛，运行期间管理器的输入新鲜度检查另行控制。

正常启动时可跳过这条独立检查，因为 `launch` 内置了同样的门禁。

## 3. 只读预检

```bash
./collectctl.sh preflight
```

所有项目都必须为 `[PASS]`。脚本会检查：

- ThinkPad 机器人网口、`192.168.123.222` 和到 G1 的路由；
- G1 ping、SSH、部署二进制和模型文件；
- 当前手模式要求的 Modbus TCP 端口；
- HTTP 相机连续取流且实测不低于 15 FPS；
- RoboticsService 和 PICO 实时时间戳；
- 没有残留 tmux、G1、导出器、相机 viewer 或手桥进程；
- 新数据集未被占用，或者续采数据集 parquet、MP4、元数据完全一致。

如果任何一项失败，不要继续启动真机。

所有子命令以退出码 `0` 表示成功，非 `0` 表示失败；终端出现 `[FAIL]` 时，即使前面已有若干 `[PASS]`，本次预检仍然失败。

`launch` 会自动先执行这一预检，因此正常流程直接运行 `launch` 即可；单独执行 `preflight` 主要用于只检查、不创建 tmux。

## 4. 启动完整 tmux 采集栈

想先查看脚本实际会执行的完整 launcher 指令：

```bash
./collectctl.sh launch --dry-run
```

正式启动：

```bash
./collectctl.sh launch
```

它会再次运行预检，然后建立：

```text
data_collection 窗口
  pane 0 左上：G1 C++ 部署
  pane 1 左下：PICO 管理器和动作可视化
  pane 2 右上：数据导出器
  pane 3 右下：相机画面

inspire_hand 窗口：Inspire DDS ↔ Modbus 桥
latency 窗口：延迟曲线
```

第一次进入 tmux 后：

1. 检查右手窗口没有 `Connection refused`、`No route to host`、Modbus timeout 或新的 DDS 发送失败。
2. 检查动作预览跟随人体，状态仍为 `robot control OFF`。
3. 检查相机画面正常。
4. 点击 pane 0，在提示 `Proceed with REAL robot deployment? [Y/n]:` 时按 Enter。
5. 等待 G1 pane 出现 `Init Done`；保持急停可触及。

tmux 操作：

```text
Ctrl+b 后按方向键    切换 pane
Ctrl+b 后按 n / p    切换窗口
Ctrl+b 后按 d        退出 tmux 但保持采集进程
```

重新进入：

```bash
./collectctl.sh attach
```

## 5. 启动策略、进入动作跟踪和录制

站在中性校正姿态：

1. 短按 `A+B+X+Y` 约 0.2 秒，然后四键全部松开至少 0.25 秒。
2. 应看到 `Calibration completed` 和 `StreamMode switch: OFF -> PLANNER`。
3. 需要全身动作跟随时，按一次 `A+X` 并完全松开。
4. 应看到 `StreamMode switch: PLANNER -> POSE`。

在另一个终端运行（或者先用 `Ctrl+b d` 脱离 tmux）：

```bash
cd /home/data-collection/GR00T-WholeBodyControl/tools/0902collect
./collectctl.sh health
```

它同时检查 G1/所需灵巧手的路由、G1 零丢包、所有进程、远端部署、最近手部错误，以及 G1 状态、PICO 和相机数据的新鲜度。任何 `[FAIL]` 都表示禁止开始或继续录制；按下物理急停并按故障恢复章节处理。首次录制前必须执行，之后建议每个 episode 开始前执行，换线、重插、重启或出现延迟后必须重新执行。

录制键与 `dunpick` 保持一致：

```text
左手 Grip + 右手 A    开始录制
完全松开后再次按       停止并保存
左手 Grip + 右手 B    丢弃当前正在录制的 episode
```

开始后导出器应显示：

```text
[Recording] 开始记录 episode N
```

停止后必须依次看到：

```text
[Recording] 结束记录 episode N
[Recording] Finished saving episode
```

没有看到保存完成前，不要杀 tmux 或导出器。不要连续重复按键。

## 6. 精简状态检查

先用 `Ctrl+b d` 脱离 tmux，再执行：

```bash
./collectctl.sh status
```

它只显示关键项：进程、PICO/模式转换、录制事件、手错误、当前 recording 状态和数据集一致性。

`status` 用于查看信息，不代表可以安全录制；是否允许录制以 `health` 的最终 `[PASS]` 为准。

## 7. 验证最新 episode 的视觉和数据

```bash
./collectctl.sh verify-latest
```

成功示例：

```text
[PASS] latest episode=3
  parquet: ... (4015 frames)
  video: ... h264 640x480 50/1 duration=80.300000s frames=4015
  longest exact duplicate run: 2 frames (0.040s)
```

metadata、parquet、视频解码帧数必须一致。脚本还会报告最长完全重复画面时长，用于发现明显冻结。

## 8. 正常停止

1. 如果正在录制，按一次左 Grip + 右 A，等待 `Finished saving episode`。
2. 按一次 `A+B+X+Y` 并完全松开，停止策略。正常情况下 PICO 管理器会退出。
3. 脱离 tmux，运行：

```bash
./collectctl.sh stop
```

如果从未启动策略、一直停在 OFF 预览，可显式确认后停止：

```bash
./collectctl.sh stop --off-preview-confirmed
```

脚本检测到仍在录制或活动状态不明确时会拒绝清理。

## 9. 急停并删除当前未完成 episode

先按下物理急停，确认机器人停止。然后执行：

```bash
./collectctl.sh emergency-stop \
  --physical-estop-pressed \
  --delete-active
```

脚本按以下顺序执行：

1. 停止 PICO 管理器、手桥和 G1 部署；
2. 给导出器时间把内存中的当前 episode 标记为 discarded；
3. 只删除位于数据集尾部且明确标记为 discarded 的 parquet、MP4 和元数据；
4. 保留此前完整 episode；
5. 修复 `info.json`、`episodes.jsonl`、`episodes_stats.jsonl` 的一致性。

它不会对未标记为 discarded 的历史 episode 执行删除，也会核对当前导出器的数据集名与 `session.env` 一致；不一致时拒绝清理。急停恢复后必须从第 2、3 步重新检查，不能直接恢复策略。

## 10. 断电、掉线或 PICO 卡住后的恢复

### 机器人中途断电或机器人网线掉线

1. 立即按下物理急停；即使机器人已经断电，也保持急停状态。
2. 如果采集栈还在运行，执行：

   ```bash
   ./collectctl.sh emergency-stop --physical-estop-pressed --delete-active
   ```

   它只会清理明确标记为 discarded 的尾部 episode；已经完整保存的历史 episode 不会删除。
3. 等机器人完全断电后，检查交换机供电、ThinkPad USB 网卡和 G1/所需灵巧手网线，再给机器人上电并等待启动完成。
4. 恢复有线连接：

   ```bash
   nmcli connection up '有线连接 1'
   ```

5. 如果机器人没有重启、网络恢复后远端仍残留旧部署进程，先执行：

   ```bash
   ./collectctl.sh remote-stop --physical-estop-pressed
   ```

6. 使用原来的数据集名重新生成续采配置：

   ```bash
   ./collectctl.sh init --dataset 09_06_pick_box_3 \
     --task 'pick up the box' --hand-mode right-only \
     --camera-url 'http://222.29.157.156:8765/stream.mjpg' --resume
   ```

7. 从 `pc-service`、PICO 重新校正并 `Send`、`launch` 完整重走；`launch` 会自动执行预检。不要从组合键步骤直接恢复。

### 只有 PICO 卡住，且机器人仍在 OFF 预览

确保没有正在录制且从未开启策略，然后执行：

```bash
tmux send-keys -t sonic_data_collection:data_collection.1 C-c
./collectctl.sh pc-service --restart
# 在 PICO 重新校正并点击 Send
./collectctl.sh pico-start
./collectctl.sh health
```

`pico-start` 会先自动执行 12 秒 PICO 检查，然后只恢复现有 tmux 中的 PICO 管理器；成功反馈为 `[PASS] PICO manager restored in robot control OFF preview`。如果策略已经启动、机器人动作异常或无法确认是否处于 OFF，不使用此局部恢复；直接物理急停并执行完整急停恢复。

## 11. 主要超参数

| 配置 | 当前值 | 含义 |
|---|---:|---|
| `DATASET_NAME` | 每次设置 | 输出目录名；新建时绝不能重名 |
| `TASK_PROMPT` | `pick up the box` 示例 | 每帧保存的任务语言描述 |
| `RUN_MODE` | `new` / `resume` | 新建数据集或续采已有数据集 |
| `HAND_MODE` | `right-only` | 只驱动右手；左手保存为合成张开状态 |
| `CAMERA_HTTP_URL` | 当前 `222.29.157.156:8765` | 单路 ego-view MJPEG 来源 |
| `DATA_FREQUENCY` | `50` Hz | parquet 和视频目标采样率 |
| `HAND_STATE_FREQUENCY` | `50` Hz | Inspire 实际角度读取/发布频率 |
| `SONIC_ZMQ_PORT` | `5556` | PICO pose/planner/manager 数据 |
| `STATE_ZMQ_PORT` | `5557` | G1 状态和 robot_config |
| `MAX_RUNTIME_S` | `3600` s | 单次真机发布授权上限 |
| `STREAMING_TIMEOUT_S` | `5.0` s | SONIC 输入兼容超时参数 |
| `LOWSTATE_TIMEOUT_S` | `0.2` s | G1 LowState 陈旧时停止控制的阈值 |
| `AUDIO_CUE_VOLUME` | `0.6` | 开始、停止、丢弃提示音量 |
| `LATENCY_SAMPLE_INTERVAL` | `0.1` s | 延迟日志采样间隔 |
| `LATENCY_HISTORY_SIZE` | `180` | 延迟窗口保留的采样点数 |

固定网络参数位于 `session.env`，当前值如下：

| 配置 | 当前值 | 含义 |
|---|---|---|
| `ROBOT_INTERFACE` | `enx00e04c36c4e8` | ThinkPad 接机器人交换机的有线网卡 |
| `PC_ROBOT_IP` | `192.168.123.222` | ThinkPad 在机器人局域网中的静态地址 |
| `G1_IP` | `192.168.123.164` | G1 地址 |
| `G1_INTERFACE` | `eth0` | G1 部署程序使用的机器人网卡 |
| `LEFT_HAND_IP` | `192.168.123.210` | 左手 Modbus 地址 |
| `RIGHT_HAND_IP` | `192.168.123.211` | 右手 Modbus 地址 |
| `HAND_PORT` | `6000` | Inspire Modbus TCP 端口 |
| `WIFI_INTERFACE` | `wlp0s20f3` | PC Service 所用 ThinkPad Wi-Fi 网卡 |

网络、模型或频率确实发生变化时，先编辑生成的 `session.env`，再运行 `./collectctl.sh config` 和 `./collectctl.sh preflight`。不要在启动后的同一个数据集内改变这些参数。

无相机采集：将 `session.env` 的 `CAMERA_ENABLED=false`。`launch` 会跳过相机检查，不启动相机预览，导出器只保存状态/动作等非图像数据，不等待图像，也不创建视频字段或 MP4；`health` 与 `verify-latest` 同样支持该模式。机器人、PICO 和手部检查仍然执行。
恢复视频采集时设置 `CAMERA_ENABLED=true`。同一数据集不能在有视频/无视频模式间切换，需另建数据集；`init` 默认恢复为 `true`，无相机采集需在初始化后再次设置。

相机接在另一台电脑上时，在 `session.env` 修改下面这一行中的 IP（端口和路径以相机电脑上的 HTTP 服务为准）：

```bash
CAMERA_HTTP_URL=http://222.29.157.156:8765/stream.mjpg
```

`CAMERA_HTTP_URL` 非空时优先使用 HTTP 相机源，无需修改备用 ZMQ 相机配置 `CAMERA_HOST` / `CAMERA_PORT`。保存后运行 `./collectctl.sh config` 核对 `camera` 行，再运行 `./collectctl.sh preflight`；检查通过后执行 `./collectctl.sh launch`。配置修改在下次启动时生效，已运行的进程不会自动切换地址。

同一相机服务仅更换电脑 IP/主机地址时，续采检查会输出 `[WARN] camera host changed` 并允许继续；协议、端口、路径等其他 URL 参数变化仍会拦截。检查不会改写已有数据集中的历史相机地址。此规则适用于同一相机和相同采集设置下的地址迁移。

以后用 `./collectctl.sh init ...` 创建采集配置时，可在原命令后加 `--camera-url 'http://新IP:8765/stream.mjpg'`；省略该参数会使用 `collectctl.sh` 中 `set_defaults()` 的默认地址，而不是保留旧 `session.env` 的地址。

模型参数保持：

```text
decoder/encoder: policy/sonic_v1_1/model
observation:      policy/sonic_v1_1/observation_config.yaml
planner:          planner/target_vel/V2/planner_sonic_trt85_static_einsum5d.onnx
motion data:      reference/example/
```

除非更换模型、相机或网络拓扑，不要修改这些值，也不要添加 `--overwrite-existing-dataset`。

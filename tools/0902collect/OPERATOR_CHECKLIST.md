# 0902collect 现场操作清单

这是日常执行版。第一次使用或发生故障时，阅读同目录的 `README.md`。任何命令出现 `[FAIL]` 都立即停止下一步；真机已启动时按物理急停。

## A. 新建一批数据

物理准备：清空机器人周围，急停拿在手中；ThinkPad USB 网卡、G1、所需 Inspire 手接同一交换机；ThinkPad 与 PICO 接同一 Wi-Fi；相机 HTTP 流已启动；最后给交换机、手和 G1 上电并等待启动。

```bash
cd /home/data-collection/GR00T-WholeBodyControl/tools/0902collect

./collectctl.sh init --date 09_06 --action pick --object box --run 1 \
  --task 'pick up the box' --hand-mode right-only \
  --camera-url 'http://222.29.157.228:8765/stream.mjpg'

./collectctl.sh config
./collectctl.sh pc-service
```

`config` 必须显示手姿态文件为 `tools/0902collect/inspire_hand_pose.json`，并输出当前 `open`、`grasp` 和 `thumb_rotate` 摘要。需要调整开合时，只在停止采集栈后修改这一个 JSON，再重新运行 `launch`。执行 `launch` 后，右手桥启动时会在保持其余五个通道现状的同时把配置中的固定拇指横向目标应用到第六通道；后续开合指令也会保持该目标。因此运行 `launch` 前就必须清空右手周围并把急停放在手边。

在 PICO/XRoboToolkit 中填写 `config` 显示的 `PICO PC Service IP`，选择身体设备、重新校正并点击 `Send`。然后：

```bash
./collectctl.sh launch
```

`launch` 会自动执行完整预检，其中包括连续 12 秒的 PICO 检查。

期望依次看到：

```text
[PASS] PICO body data is live
[PASS] all launch gates passed
Proceed with REAL robot deployment? [Y/n]
```

在 pane 0 按 Enter，等待 `Init Done`。短按并完全松开 `A+B+X+Y`，等待 `OFF -> PLANNER`；按并完全松开 `A+X`，等待 `PLANNER -> POSE`。用 `Ctrl+b d` 脱离 tmux：

```bash
./collectctl.sh health
```

只有看到下面这一行才可录制：

```text
[PASS] collection is healthy; recording may start or continue
```

```bash
./collectctl.sh attach
```

录制键：

```text
左 Grip + 右 A：开始；完全松开后再次按：停止并保存
左 Grip + 右 B：丢弃当前正在录制的 episode
```

停止录制后等待 `Finished saving episode`，再检查：

```bash
# Ctrl+b d 脱离 tmux后执行
./collectctl.sh status
./collectctl.sh verify-latest
```

## B. 续采已有数据集

把数据集名替换为真实名称，其他参数必须与原数据集一致：

```bash
cd /home/data-collection/GR00T-WholeBodyControl/tools/0902collect
./collectctl.sh init --dataset 09_06_pick_box_3 \
  --task 'pick up the box' --hand-mode right-only \
  --camera-url 'http://222.29.157.228:8765/stream.mjpg' --resume
./collectctl.sh config
./collectctl.sh pc-service
```

再次确认 `config` 显示正确的手姿态文件和预期的 `open/grasp/thumb_rotate` 摘要。

PICO 重新校正并 `Send`，然后继续：

```bash
./collectctl.sh launch
```

后续 Enter、`A+B+X+Y`、`A+X`、`health` 和录制步骤与 A 相同。

## C. 正常结束

停止当前录制并等待 `Finished saving episode`；按下并完全松开 `A+B+X+Y` 退出策略；然后：

```bash
# Ctrl+b d 脱离 tmux后执行
./collectctl.sh stop
./collectctl.sh verify-latest
```

如果策略从未启动、一直是 OFF 预览：

```bash
./collectctl.sh stop --off-preview-confirmed
```

## D. 机器人断电、掉网或动作异常

先按物理急停，再执行：

```bash
./collectctl.sh emergency-stop --physical-estop-pressed --delete-active
```

检查线缆和交换机，机器人重新上电并启动后：

```bash
nmcli connection up '有线连接 1'
./collectctl.sh init --dataset 09_06_pick_box_3 \
  --task 'pick up the box' --hand-mode right-only \
  --camera-url 'http://222.29.157.228:8765/stream.mjpg' --resume
./collectctl.sh pc-service --restart
```

PICO 重新校正并 `Send`，然后执行 `./collectctl.sh launch`；它会自动完成 PICO 检查和全部预检。不要直接按组合键恢复。

## E. 三条铁律

1. `preflight` 或 `health` 有任意 `[FAIL]`，不录制。
2. 没看到 `Finished saving episode`，不关导出器、不杀 tmux。
3. 新采集不用旧目录名；续采必须显式写 `--resume`，永远不加覆盖参数。

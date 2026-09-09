# ThinkPad REF 夹爪桥接（2026-09-09）

该入口复用 ThinkPad 现有 REF USB 驱动，订阅机器人原有 PICO 右扳机 ROS 2 消息，底层使用 Fast DDS；不改成 Unitree 原生 rt/lowstate 消息，不启动身体、灵巧手或 PICO 服务。

## 接线与进程

- PICO 网线接机器人；XRoboToolkit PC Service 和 PICO 解析仍由机器人运行。
- Inspire 灵巧手仍由机器人现有控制链路负责。
- REF USB 和 RealSense USB 接 ThinkPad。
- ThinkPad 与机器人通过有线网互通；本桥接只控制 REF，数采入口另行运行。

订阅主题：

```text
ControlPolicy/upper_body_pose
```

消息类型是 std_msgs/ByteMultiArray，内容是 MsgPack 字典；消费 right_trigger，阈值与原桥接相同，严格大于 0.5 表示按下。当前 grab_beaker 使用机器人现有配置快照：open=0.4，pressed=0.1。不能沿用另一个推理项目的 open=1/closed=0 代替本数采任务。

## 现有驱动与独立环境

以下每行是一个完整路径：

```text
/home/data-collection/projects/inspect-robots/.deployment/ref-control/agibot_ref.py
/home/data-collection/projects/inspect-robots/.deployment/ref-control/ref_control.py
```

这两个原文件未修改。仅导入其 RefClient、PyUsbTransport 和固件序列号定义；没有调用 inspect-robots 的手臂或混合手部初始化。

本 worktree 的 .venv_ref_bridge 由 uv 创建，使用已有 decoupled_vla_collection Conda Python，并通过 system-site-packages 复用 ROS/MsgPack 依赖，仅在新环境安装 pyusb==1.3.1。未修改原 Conda 环境或 inspect-robots 环境。环境因此依赖该 Conda 环境继续存在，不是可独立搬运的完整环境。

独立环境的创建/复验命令如下；先进入本 worktree，再单独执行第二行：

```bash
cd /home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-receive-time-20260907
bash scripts/setup_thinkpad_ref_bridge.sh
```

本机已完成上述步骤。脚本默认使用本项目 .tools/ref-bridge/uv；该工具副本和虚拟环境未提交 Git。其他机器需先提供已有 uv 可执行文件，通过 UV_BIN 指定，BASE_PYTHON 指向其已配置 ROS 的项目 Conda Python。不要向 base 安装依赖。

## 启动

以下命令均在上述 worktree 内执行，每个代码块的命令保持一行。

只检查依赖、配置和 libusb 可用性；不创建 ROS 节点、不打开 USB：

```bash
bash scripts/run_thinkpad_ref_bridge.sh --hand-task grab_beaker --preflight
```

订阅真实 PICO 消息但使用模拟 USB 传输，15 秒后退出，不驱动夹爪：

```bash
bash scripts/run_thinkpad_ref_bridge.sh --hand-task grab_beaker --dry-run --duration 15
```

实际夹爪控制：

```bash
bash scripts/run_thinkpad_ref_bridge.sh --hand-task grab_beaker
```

实际模式等待唯一发布者和新鲜的“右扳机松开”消息，才打开 USB，检查 schema、固件序列号、故障码并执行 prepare。初始化完成后再次等到新鲜的松开状态，再发送张开目标；按下右扳机切换到 pressed 目标。目标未变时不重复发送。

该命令不启动机器人控制。如果没有原 PICO 解析程序发布上面的主题，程序只等待，不会操作夹爪。PC Service 单独启动并不等于扳机主题已经在发布。

不要同时在机器人上运行原 REF USB 桥接。机器人旧综合 launcher 的 ref-gripper-bridge 标志还参与“仅左 Inspire”的配置推导，因此不能把“删除这一个标志”当成完整的机器人端启动改造，也不能用 left-hand-only 替代它而忽略右臂/PICO 语义变化。本次交付是 ThinkPad REF 桥接，不修改机器人综合 launcher；身体/左手/PICO 需由已经正确配置的控制链路提供。

## DDS 有线配置

启动包装脚本只在当前进程设置 Fast DDS 环境，使用 thinkpad_ref_fastdds.xml 把 UDP 接口限制到 ThinkPad 有线地址 192.168.123.222。没有修改系统路由、网络配置、SSH、防火墙或其他进程环境。

ROS_DOMAIN_ID 沿用启动终端环境，未指定时通常为 0，必须与机器人发布进程一致。ROS_LOCALHOST_ONLY 默认 0。自定义 RMW_IMPLEMENTATION 或 FASTRTPS_DEFAULT_PROFILES_FILE 时沿用用户值；使用其他 RMW 时此 Fast DDS XML 不生效。

如果 ThinkPad 有线地址改变，应更新本配置或提供正确的 Fast DDS profile，不能期待自动回退到 Wi-Fi。接口白名单使用 Fast DDS 官方配置格式：
[官方接口白名单说明](https://fast-dds.docs.eprosima.com/en/v2.6.11/fastdds/transport/udp/udp.html)。

## 断流、故障与关闭

- 默认本机消息年龄超过 0.3 秒、扳机无效或发布者不唯一：停止发送新目标，保持最后一次目标；这不等于急停，也不主动开合夹爪。
- 恢复消息后，需要先松开右扳机再继续，避免恢复瞬间执行一直按住的目标。
- 发现发布者身份变化：退出，需为新的发布源重新启动。
- USB 初始化/prepare/故障检查失败、写入或 ACK 失败：记录错误并退出；不自动清错、不尝试其他固件命令。
- Ctrl+C 退出仅释放本进程 USB/ROS 资源，不发送张开或归零指令。由于驱动关闭不是夹爪急停，最后目标可能继续由硬件执行。

## 日志与采集时间

每次启动自动生成一个独立 JSONL 日志，路径会在终端打印，默认位于本 worktree 的 .runtime/ref_bridge。也可以通过 log-file 指定一个尚不存在的文件，不覆盖旧日志。

日志记录会话、任务配置、扳机接收单调时间、命令请求时间、ACK 时间、目标值、断流状态和错误。与 ThinkPad 采集程序使用同一主机 monotonic 时间，可后续关联。

commanded_position 是收到 ACK 的目标值；measured_position 明确为 null。没有确认当前 REF USB 固件提供独立实测开口，不能把目标值、ACK 或函数输入端点当作实测位置。dry-run 日志带 dry_run=true，其 ACK 为模拟结果。

当前日志独立保存，尚未自动合入 LeRobot 行或其 41 维 action.wbc；需要 REF 动作标签时应使用该日志按单调时间关联，不能把旧 schema 的右侧六维 Inspire 字段当作 REF 实测开口。

## 验证结果

2026-09-09 完成：

1. 新 uv 环境和启动脚本按实际参数执行 preflight，原驱动导入与 libusb backend 检查通过。
2. 回环 ROS 2/Fast DDS、真实 ByteMultiArray/MsgPack 消息、原 SDK DryRunTransport 全链路测试通过；目标序列 0.4→0.1→0.4→0.1。
3. 初始按住、阈值边界、过期/负年龄、NaN、多发布者、断流恢复必须先松开等检查通过。
4. ACK 故障注入：请求有记录，但没有虚假的 command_acked，也不更新 last_acked_command。
5. 启动包装脚本加载实际有线 profile，以 dry-run 运行并正常退出。
6. 机器人发布专用测试主题，ThinkPad 通过有线 profile 接收，得到 0.4→0.1→0.4。测试使用独立 ROS domain 207 和私有测试主题，不发布机器人运动指令。
7. Python 语法、Shell 语法和 Git diff 检查通过。临时测试脚本和自产测试日志完成后删除。

本轮没有打开真实 REF USB 控制连接，没有 prepare、开合或手臂动作，没有切换两台机器原有工作目录的分支。未验证实际 PICO 扳机实时联控、实际夹爪运动与长时稳定性。

## 代码

- [桥接入口](../../../gear_sonic/scripts/run_thinkpad_ref_bridge.py)
- [启动包装](../../../scripts/run_thinkpad_ref_bridge.sh)
- [环境配置](../../../scripts/setup_thinkpad_ref_bridge.sh)
- [固定依赖](../../../install_scripts/requirements_thinkpad_ref_bridge.txt)
- [烧杯夹爪目标](../../../gear_sonic/config/data_collection/thinkpad_ref_gripper_tasks.json)
- [有线 DDS profile](../../../gear_sonic/config/data_collection/thinkpad_ref_fastdds.xml)

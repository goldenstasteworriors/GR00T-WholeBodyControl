# ThinkPad 本机接收时间采集程序

实现日期：2026-09-08。只记录入口；不启动 WBC、PICO、导航、Inspire 或 REF 夹爪控制进程，不做对时和延迟补偿。当前支持一台直接 USB 连接 ThinkPad 的 RealSense RGB 相机，以及通过有线局域网可达的机器人 decoupled ROS 状态。

## 已实现的行为

- RealSense RGB 640×480、30 FPS，读到帧后立即打 ThinkPad monotonic 时间戳，再复制像素；RGB 原样写入视频，不通过 JPEG/ZMQ 中转。
- 机器人状态在笔记本 ROS 回调入口打同一主机时间戳。只有收到一个来源的状态且字段有效，才允许开始录制。
- 默认 50 Hz 取最新快照；记录相机原帧号和接收时间，重复视频帧不刷新时间戳。
- 复用 41 维状态/动作及原末端、IMU 等 LeRobot 字段；另写 receive_time.jsonl，保存逐行 tick、接收时间、图像帧号、接收年龄、原始 43 维 q/action、episode 开始结束事件。
- 不从机器人时间推算延迟。receive_delta_ms 只表示两路数据抵达笔记本的时间差。
- 默认任一路本机数据年龄或记录循环间隔超过 100 ms，则保存当前有效片段并停止录制，等待用户重新开始。这是数据新鲜度阈值，不是同步精度保证。
- 相机超时/断开会报错并退出；不会自动重连后把新帧接入旧 episode。数据年龄阈值已触发时先保存此前有效片段。其他异常记录 session_error；出现该事件的未完成片段不得当作已验证数据。
- 启动要求新数据集目录；禁止覆盖既有数据集。相机序号回退会终止本次进程。原始设备时间仅作追溯。
- 输入使用有锁的单槽最新值，没有累积无限队列；50 Hz 消费未覆盖的高频状态不逐条写盘。它是最新值快照记录器，不是底层所有采样的无损日志。

## 启动前条件

机器人仍需由原有控制流程提供 G1Env/env_state_act 状态和 WBCPolicy/robot_config 配置服务。笔记本不会重复启动任何机器人控制。不要在笔记本运行原综合 tmux launcher 代替此入口。

ROS domain 与有线 DDS 发现必须可达；有网线不自动代表 ROS 跨机可达。沿用实际机器人使用的 ROS_DOMAIN_ID / DDS 接口设置。本次未改动两台机器网络或正在运行的进程，也未验证真实有线 DDS 链路。配置服务默认等候 30 秒，失败会退出；不要据此切换机器人分支或停止已有程序。

RealSense 不应被另一个程序占用。此入口直接占用相机，不同时运行 composed_camera 服务。多台 RealSense 时必须显式指定序列号。

## ThinkPad 启动命令

先执行第一行进入独立 worktree，再执行第二行。第二行保持整行，无需手动换行。使用已验证的项目 Conda Python，不需要安装依赖。

```bash
cd /home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-receive-time-20260907
PYTHONDONTWRITEBYTECODE=1 /home/data-collection/miniconda3/envs/decoupled_vla_collection/bin/python -m gear_sonic.scripts.run_thinkpad_receive_collection --task-prompt "move the beaker from left" --dataset-name thinkpad_beaker_20260908_01 --root-output-dir outputs/thinkpad
```

每次重新启动请换一个尚不存在的 dataset-name。同一次进程可以连续录制多个 episode。数据写在该 worktree 的 outputs/thinkpad 下。

安全预检使用同一条命令，在末尾追加以下参数（不单独执行）；它仅检查导入、模型与抓握配置，不打开相机，不初始化 ROS：

```text
--preflight
```

多相机时，在启动命令末尾追加以下参数片段，替换占位符，保持同一行：

```text
--realsense-serial <RealSense序列号>
```

## 本地按键

在运行采集程序的终端输入单个字符并按 Enter：

- c：开始记录；再次 c 保存结束。
- x：保存当前 episode 并在 LeRobot metadata 中标记 discarded，不删除文件。
- q：保存当前 episode 并退出。
- Ctrl+C：正常情况下保存当前片段并退出。写盘本身失败会报错，不能保证失败片段完整。

这些按键不会广播到机器人，也不使用原机器人键盘/manager_state 控制频道。

## 手部与 REF 夹爪语义

默认抓握配置是机器人在 2026-09-08 的 grab_beaker 配置快照，保存在 thinkpad_grab_beaker_pose.json。左手 open 为 [1,1,1,1,1,0]，pressed 为 [0.37,0.37,0.37,0.37,1,0]；本任务未单独配置 grip/grip_pressed，因此按机器人现行规则回退到 open/pressed。

七维历史手指命令到六维手部指令的转换与机器人现行 command_sender.py::legacy_dex3_to_inspire 的四状态判定一致。原始动作另外保存在关联文件中，可检查推算过程。更换任务必须通过 inspire-hand-pose-config 提供对应六维姿态配置，不要沿用烧杯配置记录其他任务。

REF 夹爪仍由机器人桥接服务控制；现有 41 维格式并不含独立的 REF 夹爪实测开口字段，右侧六维手部字段不能作为 REF 夹爪实测值。本次保持现有数据约定，没有新增独立 REF 夹爪测量。若需要该实测量，需要机器人发布相应反馈并增加数据字段。

不同频率的手部反馈在合并消息中的独立年龄当前未知，不宣称手部反馈与身体同时采样。

## 验证记录

在 ThinkPad 项目环境完成：

1. 相同入口、任务参数、输出参数加 preflight：模型 43 关节、27 个 schema 字段，成功；未打开硬件。
2. 模拟输入驱动实际 collector 与 Gr00tDataExporter：6 行 parquet、6 帧可解码 MP4、41 维状态和动作、逐行时间关联及重复帧时间保持，通过。
3. 断流陈旧片段停止、discard 标记、多发布者拒绝开始、ROS shutdown 保存，通过。
4. 缺失字段、NaN、负年龄、超龄检测，以及烧杯四状态动作映射，通过。
5. Python AST 与 Git diff 格式检查，通过。临时测试脚本与自产测试数据测试结束后删除。

未验证：真实相机帧流、有线跨机 ROS 发现、真实状态与相机联采、实际同步误差、长时磁盘性能。为保持当前运行程序，本轮不执行硬件采集。

## 文件入口

- [仅记录 CLI](../../../gear_sonic/scripts/run_thinkpad_receive_collection.py)
- [本机输入和记录循环](../../../gear_sonic/utils/data_collection/thinkpad_receive.py)
- [复用的 exporter](../../../gear_sonic/scripts/run_decoupled_vla_data_exporter.py)
- [烧杯手部配置](../../../gear_sonic/config/data_collection/thinkpad_grab_beaker_pose.json)
- [最初方案](thinkpad_wired_receive_time_collection.md)

## 2026-09-09 REF 桥接

ThinkPad 上的 PICO→REF USB 独立桥接已实现，见 [REF 桥接说明](thinkpad_ref_bridge.md)。它另存同机单调时间日志，尚未自动合入 LeRobot 字段；实测开口仍不可用。

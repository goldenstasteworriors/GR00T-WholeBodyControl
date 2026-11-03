# NaVILA → ThinkPad Balance/Walk → G1

本次接入复用 ThinkPad 桌面《双Dex1三相机采集完整流程.txt》对应工程，原采集目录不改。独立分支为 `codex/navila-g1-01a19ca8`，基于 `c1495a7`。会话 `01a19ca8-ac6e-7460-a4da-6a193efe13ca`，任务轮次 9。

## 已验证和未验证

2026-11-03 实测：

- G1 有线地址 192.168.123.164，ThinkPad 网卡 enx00e04c36c4e8，地址 192.168.123.222。
- 机器人已有相机服务端口 5555；订阅 `ego_view`，RGB 640×480，另有左右腕相机。没有重启机器人服务。
- 真实相机 → ThinkPad → SSH 中转 → server02 NaVILA → ThinkPad：两次往返 1.0625 / 1.1311 秒；这是图像编码后网络加推理耗时，不含全部采集延迟。模型输出左右转都能严格解析，未发送运动指令。
- 按下方烧杯指令实际运行一次，约 2.012 秒返回 `The next action is turn left 45 degree.`，只记录未执行。这不是烧杯停靠成功的证据，也未据此启用自主行走。
- 隔离 ROS domain 178 的模拟接收端：中位约 49.1 Hz，发布器突然消失后约 0.188 秒归零；图像断流锁定停止；非法动作、超限速度、过期/重放消息被拒绝。测试没有连接机器人执行器。
- 原 Balance/Walk ONNX 在 CPU 上加载、前向计算通过。零输入只验证加载/计算，不验证运动能力。
- 实际控制脚本以 `--read-only` 运行：读取真实 29 维关节状态，计算 29 维目标，到达 `ready`，`use_policy_action=false`、`diagnostic_read_only=true`、导航速度为零；不切换官方控制模式，不发位置/阻尼/手部命令。
- 尚未执行 `--real` 的带电机写入控制，未使能站立/行走，未完成实机转向、前进、避障、烧杯停靠验收。`--real` 与已测试的只读路径相同，仅去掉只读标志；其实际运动效果仍待现场验收。

验证记录（ThinkPad）：`.runtime/navila/live-inference.jsonl`、`beaker-readonly.jsonl`、`verification.json`、`readonly-control.log`、`readonly-status.json`、`.runtime/collection_damping_stop.json`。临时测试脚本在测试后删除，不作为部署依赖。

## 链路与控制职责

```text
G1 ego RGB --ZMQ 5555/网线--> ThinkPad camera/history
                                  |
                              TCP 127.0.0.1:54321
                                  |
                          本机 SSH 双隧道中转
                                  |
                         server02 NaVILA / RTX 4090
                                  |
                       文本动作：前进 / 左右转 / 停止
                                  |
ThinkPad bridge --ROS domain 73, 50 Hz--> 带到期检查的 WBC
                                  |
                   Balance / Walk ONNX --Unitree DDS--> G1
```

NaVILA 不输出关节动作，也不替换你的低层。接口为 `navigate_cmd=[vx,vy,wz]`，单位 m/s、m/s、rad/s；正 wz 为左转。控制器根据速度模长选择 Balance 或 Walk。

本机中转复用既有 SSH 身份，不复制私钥。**本机必须持续开机且能连接两端**。ThinkPad 目前没有 server02 直连 SSH 配置；若未来取消本机中转，需另行配置两端网络/SSH。所有推理隧道端口仅绑定 loopback，没有对外开放 NaVILA。

server02 模型服务由上一轮部署的 tmux `navila-server02` 运行。代码 `/home/user/ykj/project/NaVILA`，模型和环境在 `/data/ykj/navila`，本次没有改动它的驱动、环境或权重。

## 启动：只读推理

以下每个代码块都是一条完整命令，不需要手动换行；不同代码块分别在指定机器执行。

1. 本机启动隧道（已按此命令实测两个 pane 正常）：

```bash
bash /home/ykj/navila-g1-link/deploy/navila/local_relay.sh start
```

查看或停止隧道：

```bash
bash /home/ykj/navila-g1-link/deploy/navila/local_relay.sh status
```

```bash
bash /home/ykj/navila-g1-link/deploy/navila/local_relay.sh stop
```

`relay_ssh_config` 是本机实际部署配置，包含现有 HostKeyAlias，留在部署目录并被 Git 忽略；不包含密钥。仓库里的 relay 脚本需要同目录该配置才能使用。隧道中断时不会自动无限重启，检查 pane 报错后 stop/start。

2. ThinkPad 只读请求，不创建 ROS 指令发布器：

```bash
bash /home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/run_bridge.sh --instruction "Find the beaker and stop in front of it." --steps 1
```

使用现有 Conda `decoupled_vla_collection`；无需安装新库。图像按历史采样 7 帧加当前帧，不足 8 帧前补黑帧。一次进程代表一个任务，历史只驻内存。`--steps` 默认 1，上限 100。

3. ThinkPad 只读验证关节状态与 WBC（已按脚本实测）：

```bash
bash /home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/run_control.sh --read-only
```

结束只读检查可在其终端按 Ctrl+C。不要同时运行原采集控制器；脚本发现已有控制/遥操作进程会退出。只读控制器不会满足 bridge 的执行握手。

## 现场实机阶段（尚未执行）

当前相机画面主要是近处实验桌面，地面/通道覆盖不足。先确认前方无遮挡并调整视角，现场有人负责 G1 支撑与急停。模型输出不能充当避障传感器。

关闭只读控制器和原采集/遥操作进程。在 ThinkPad 终端 A 启动：

```bash
bash /home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/run_control.sh --real
```

**该命令会退出官方运动模式，并开始关节初始化；初始化本身就有动作。** 保留原采集工程的初始化逻辑，关闭 startup T pose 和手部控制，不启动 PICO、夹爪或数据导出。导航期间只设置底盘速度，不接管上肢姿态。等待打印 `ready` 后，在终端 A 按 `]`，才启用原 Balance 站立策略。

第一轮只执行一个受限导航段，在 ThinkPad 终端 B 启动：

```bash
bash /home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/run_bridge.sh --instruction "Find the beaker and stop in front of it." --execute --steps 1
```

终端 A 的 `d` 是原控制器的阻尼急停；`]` 在 running 状态也会切入阻尼。停止 bridge（Ctrl+C）只归零导航速度，低层保持站立。任务完成同样是零速站立，**不是阻尼/断电**。低层控制器的 Ctrl+C 会走原阻尼退出逻辑。没有 PICO 进程，因此这个导航入口使用键盘使能，不依赖采集流程的四键组合。

## 延迟、限幅和故障行为

- 模型推理期间发布零速；执行完一段后零速等待 0.5 秒，再观察/推理。高层可以约 1 秒一次响应，底层 50 Hz 不等待网络。
- 初始限幅：前进 0.15 m/s，每段最多 25 cm；转向 0.25 rad/s，每段最多 15°。原模型的 50/75 cm、30/45°请求会截短，并在 JSON 日志保留原始请求和实际限幅。
- 距离/角度按时间估计，当前真实 `BodyStateProcessor` 没有订阅导航里程计（odostate 只用于仿真）。这不是位置闭环；不能承诺 25 cm/15°的实际误差，也不能承诺烧杯前精准停靠。
- 只接受明确的前进/左右转/停止文本；未知、解释性文本、超出白名单的动作直接报错退出，不默认前进。
- 仅 ThinkPad 发布命令，附同机 monotonic 时间，最长 0.25 秒有效。控制器每一周期检查，过期直接用零速绕过旧速度插值。不会因重连重放旧动作。
- 相机 0.5 秒无新源时间戳、控制状态 0.25 秒不新鲜、竞争发布器、控制模式离开 running：锁定故障并归零，需要重启任务。
- 推理超时默认 8 秒；用于执行的观测到动作超过 4 秒即拒绝。等待期间始终零速。
- 接收端检查只在 `NAVILA_NAVIGATION=1` 的专用控制脚本启用，原采集入口不改变。
- 该超时机制针对高层/网络/bridge 故障；它不替代 G1 对底层控制进程崩溃、DDS 中断或物理碰撞的保护。
- G1 与 ThinkPad 的墙钟存在约数分钟偏差，没有修改系统时间。相机新鲜度用源时间戳是否推进 + 本机单调时钟，不把跨机器墙钟差当作传输延迟；端到端相机捕获年龄没有独立标定。

## 代码位置

- `deploy/navila/bridge.py`：相机接入、8 帧历史、TCP协议、动作白名单、50 Hz 发布、状态握手。
- `deploy/navila/navigation_lease.py`：时间戳、速度范围、动作期限和停止检查。
- `decoupled_wbc/control/main/teleop/run_g1_control_loop.py`：专用导航模式，只读状态标记与接收端期限检查。
- `decoupled_wbc/control/policy/g1_decoupled_whole_body_policy.py`：将已检查速度直接传到原低层策略，不经过旧速度插值。
- `decoupled_wbc/control/policy/g1_gear_wbc_policy.py`：原 Balance/Walk 选择与 ONNX 前向，不修改。
- `deploy/navila/run_bridge.sh` / `run_control.sh`：复用既有环境的启动脚本。
- `deploy/navila/local_relay.sh`：本机两段 SSH 隧道。

后续若要可靠完成“停在烧杯前”，需在实际导航相机视角下采集/评估动作，加入可用的位姿/深度或目标距离反馈，并验证停止距离。当前交付是可运行的软件连接和只读硬件验收，不能据此宣称该任务已闭环成功。

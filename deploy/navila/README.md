# NaVILA → ThinkPad Balance/Walk → G1：人工急停版

本版按用户要求移除了上一版新增的导航防护措施，仅增加两端人工阻尼急停入口。原 Balance/Walk 权重、原控制器的初始化、人工站立使能、关节处理和阻尼逻辑保持原样。

独立工程：/home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link

分支：codex/navila-g1-01a19ca8

会话：01a19ca8-ac6e-7460-a4da-6a193efe13ca，任务轮次 11。

## 两端急停

以下代码块各是一条完整命令，不需要手动换行。

**在本机执行：**

```bash
bash /home/ykj/navila-g1-link/deploy/navila/estop_local.sh
```

**在 ThinkPad 执行：**

```bash
bash /home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/estop_thinkpad.sh
```

**在 ThinkPad 的控制器终端：按 d。**

两端脚本对同一个控制器发送 SIGUSR1，直接复用原 DampingShutdown.request_estop → env.enter_damping；不经过 NaVILA/server02、不等待推理。控制器进入阻尼并保持该状态，程序不退出；这不是导航零速站立，也不是断电。

成功时输出 JSON 中 state=damping_latched。该回执表示控制器已执行进入阻尼的调用，不等于传感器测得机器人已经静止。diagnostic_read_only=true 表示只是只读测试，没有发送电机指令。脚本会等待本次回执，失败返回非零，不把“SSH 命令发出”误报成急停完成。

本机脚本依赖本机到 ThinkPad 的 SSH 连接；ThinkPad 本地脚本及控制窗口的 d 键不依赖该连接或 server02。两个脚本针对本目录 run_control.sh 启动并登记的控制器，不是通用的 G1 官方控制程序急停。原采集工作目录没有改动。

本机 ConnectTimeout=3 与急停回执等待 3 秒只用于报告人工急停请求的通信结果，不是机器人运动超时规则。PID/start-time 检查只用于定位正确控制进程，避免把信号发给别的进程。

## 本版删除的内容

- 删除 navigation_lease.py，控制入口与全身策略恢复至原 c1495a7 版本；不再附加指令有效期、接收端超时归零或速度覆盖。
- 删除相机 0.5 秒、控制状态 0.25 秒新鲜度检查和断流故障锁定。
- 删除推理 8 秒超时、观测到执行 4 秒限制和启动等待 120 秒限制。
- 删除前进最多 25 cm、转向最多 15°的截短，以及额外的速度范围检查。
- 删除离散距离/角度白名单；解析格式支持任意数值，如 125 cm、90°，不再限于 25/50/75 和 15/30/45。
- 删除竞争发布器检查、额外控制进程互斥和有线 carrier 预检。
- 删除 bridge 异常退出时自动发布零速的逻辑。关闭 bridge 不作为急停方法；人工急停用上面的两个脚本或 d。

保留的动作文本解析、JSON/TCP 收发和原控制器行为属于接口本身。无法解析的文本会报错，不编造一个运动动作。相机沿用最后取得的画面，不判断其是否过期。发生网络或进程异常时没有本次新增的自动停止机制。

## 动作执行

使用原 ROS topic ControlPolicy/upper_body_pose，发布 navigate_cmd=[vx,vy,wz] 和 target_time，由原上层插值/低层策略接收。单位为 m/s、m/s、rad/s。

默认前进速度 0.5 m/s，转向速度 π/6 rad/s；通过 --linear-speed / --angular-speed 可自行指定，不做额外限幅。模型输出 75 cm 就按完整 75 cm 计算执行时长，45°也按完整 45°计算，默认均为 1.5 秒。仍是按时间估计距离/角度，不是导航位姿闭环。

每段动作执行时约 50 Hz 发布指令，段结束发一次零速，再采图/推理下一段；模型输出 stop 时也发零速。这是完成动作的语义，不是超时保护。没有独立的零速心跳或强制 0.5 秒停顿。收到明确的 damped 状态会结束当前导航任务，属于人工急停效果。

## 链路和启动

G1 相机 ZMQ 5555 → ThinkPad → 本机 SSH 中转 → server02 NaVILA → ThinkPad ROS → 原 Balance/Walk → G1。

G1 地址 192.168.123.164；ThinkPad 有线网卡 enx00e04c36c4e8；ego_view 为 RGB 640×480。NaVILA 输入保持 8 帧历史，不足时前补黑帧。环境仍是现有 Conda decoupled_vla_collection，没有安装依赖或修改驱动。

server02 仍使用上一轮的部署：代码 /home/user/ykj/project/NaVILA，环境/权重 /data/ykj/navila，tmux navila-server02。当前通信仍依赖本机中转在线。

本机启动中转：

```bash
bash /home/ykj/navila-g1-link/deploy/navila/local_relay.sh start
```

查看中转：

```bash
bash /home/ykj/navila-g1-link/deploy/navila/local_relay.sh status
```

若中转断开，分别执行 stop 和 start。relay_ssh_config 是已有部署配置，Git 忽略，不包含密钥。

ThinkPad 只读推理：

```bash
bash /home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/run_bridge.sh --instruction "Find the beaker and stop in front of it." --steps 1
```

ThinkPad 只读控制诊断（不切换模式、不写电机）：

```bash
bash /home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/run_control.sh --read-only
```

ThinkPad 真实控制启动，会退出官方运动模式并进行关节初始化：

```bash
bash /home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/run_control.sh --real
```

等待 ready 后，在控制器终端按 ] 启用原 Balance 站立。导航启动如下；--steps 决定本次最多执行多少段，未指定时为 1：

```bash
bash /home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/run_bridge.sh --instruction "Find the beaker and stop in front of it." --execute --steps 1
```

急停后的恢复沿用原流程：damped 状态按 ] 重新初始化，ready 后再次按 ] 站立；重新启动导航任务。没有自动恢复行走。

## 验证范围

2026-11-03：

- 本版真实相机推理复测约 1.186 秒成功，未发送移动指令。
- 上一版真实相机到 server02 推理已测，约 1.06–2.01 秒；恢复 SSH 后请求也通过。当前相机主要朝向实验桌面，烧杯停靠尚未闭环验证。
- 本版验证完整 75 cm / 45°时长都是 1.5 秒，125 cm / 90°可解析；原插值在不接收新指令时保持目标，没有新增超时归零。
- 使用同一个真实 G1 只读控制进程，分别执行 ThinkPad 急停脚本和本机急停脚本，两次都获得当前请求的 damping_latched 回执。ROS 状态确认 damped 且 use_policy_action=false。
- 测试没有发出实际电机动作；没有进行实机行走或带电机写入的急停效果测试。实际阻尼发送复用的是原工程已存在的入口。
- 验证记录位于 .runtime/navila/manual-estop-unit.json、estop-thinkpad-result.json、estop-status.json、manual-estop-readonly.log；本机回执在 /home/ykj/navila-g1-link/runtime/estop-local-result.json。
- 旧 verification.json 中的 watchdog/限幅结果仅代表上一版，不适用于本版。

## 文件

- deploy/navila/bridge.py：取消附加限制，使用原导航接口。
- deploy/navila/estop.py：控制器登记、人工 SIGUSR1 请求和回执。
- deploy/navila/estop_local.sh：本机 SSH 急停入口。
- deploy/navila/estop_thinkpad.sh：ThinkPad 本地急停入口。
- deploy/navila/run_control.sh：配置急停登记目录，保留原控制启动参数。
- decoupled_wbc/control/utils/damping_shutdown.py：仅增加可选的登记和急停回执，不改变原阻尼逻辑。
- 原 run_g1_control_loop.py 和 g1_decoupled_whole_body_policy.py 已恢复；navigation_lease.py 已删除。

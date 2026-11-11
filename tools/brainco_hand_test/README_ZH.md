# 强脑 Revo2 双手同步测试

通过宇树官方 DDS 话题读写灵巧手，没有身体控制接口。需要同一 DDS domain（默认 0）、同一网卡上的官方强脑服务提供两只手的状态。

先停止正在运行的 `example_brainco_hand`、网页调手和其他手部命令发送程序。同一串口只运行一个服务。现场确认手中无物、手指周围有空间。

## 机器人终端 A：保持官方服务运行

逐行执行，原来的单手服务也要先退出，避免重复打开串口：

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl
sudo ./external_dependencies/brainco_hand_service/bin/brainco_hand_server --network_interface eth0
```

应同时出现 left hand bound 和 right hand bound。自动扫描服务识别后可能先张手。此前识别过左手 ttyUSB2、右手 ttyUSB1，但以本次日志为准。

## 机器人终端 B：先做只读检查

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl
bash tools/brainco_hand_test/run_both_hands.sh --network eth0
```

首跑会编译，需要已有 c++、Unitree SDK2，或项目中完整的 vendored Unitree SDK2。不会下载或安装软件。默认仅订阅反馈；需左右各收到至少三帧且反馈年龄小于 0.5 秒才通过。未收到反馈时不得绕过检查。

## 双手同步动作一次

```bash
bash tools/brainco_hand_test/run_both_hands.sh --network eth0 --move
```

两手依次同步：从当前姿态缓慢张开 → 握拳 → 张开。每段默认三秒，目标到达后保持一秒，总计约十二秒。DDS 左右消息顺序发送，属于同一控制周期（50 Hz），并非硬件严格同时触发。

使用官方推荐的归一化 speed=1，通过缓慢改变 q 来限制目标变化速度。每个阶段检查实际六指 q 与目标偏差（容差 0.15）；未到目标则不进入下一阶段。这个检查不是接触力限制或紧急停止功能。

按 Ctrl+C 停止发命令；任一手反馈陈旧、发送失败也会退出。退出时手可能保持最后目标，不保证自动张开，也不会停止独立运行的官方服务。正常结束后两手应张开。

本机用回环 DDS 验证默认不发送、缺少单侧反馈拒绝动作、双手闭环动作和陈旧反馈退出。未由本程序完成真实双手运动验收；现场按以上流程验证。不要同时运行 SONIC 采集。

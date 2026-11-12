# pku4090：G1 Dex1 双夹爪 JAR 部署

工作目录为 `/home/yanjun/projects/JAR_G1_Deploy_20261112`。模型、tokenizer、冻结的模型源码及本目录部署代码均从私有 ModelScope 仓库 `ykjykj/JAR-G1-Lab-UpperBody-Observation-20261111` 下载。
`bundle/export_manifest.json` 列出已完成操作和权重SHA256；后续训练完成的操作由同步服务继续下载。
独立uv环境位于 `.venv`，下载用的轻量环境位于 `.venv-download`。不修改旧JAR、Conda环境、CUDA或显卡驱动。
`install.sh` 按官方SDK说明将CycloneDDS 0.10.5编译到本项目runtime中，再安装Python绑定0.10.2和SDK源码。
SDK及CycloneDDS源码归档同样来自ModelScope；原许可证保留。重新安装时执行以下完整一行命令：

```bash
bash bundle/deployment/run.sh install
```
Ubuntu缺少Python开发头文件时，安装脚本只把官方dev包解压到runtime，不安装或升级系统Python。
下载服务可读取 `runtime/download-interface` 中的网卡名，仅为自身连接选择出口，不修改系统路由。

## 输入、输出与运行方式

输入顺序固定为左臂7关节、右臂7关节、左Dex1、右Dex1，角度单位rad。对应原 `observation.state[15:31]`。
相机顺序为ego_view、left_wrist、right_wrist，RGB，240×240；历史为30Hz下 `[t-24,t-16,t-8,t]`，启动时去重。
在线历史按这些相对时间选最近的因果帧，保留实际时间间隔；每路相机独立检查时间戳是否持续更新。
归一化和英文指令直接读取各操作的 `observation_representation.json`，不读取 action 或初始框。

权重预测未来32帧的观测位置。运行时可以只生成前1帧的16个token；这与完整512-token生成的首帧严格对应，
不是重新训练或改变输出定义。benchmark同时检查1/8/32帧的前缀一致性并记录真实延迟。
推理在独立线程运行，50Hz执行线程对参考位置限速、检查状态时效；旧预测保持上一安全参考，旧机器人状态终止输出。
这是一种慢速参考位置闭环，不能把完整512-token生成速度误称为30Hz控制性能。

## 推荐的初始化、站立与重力补偿流程

本流程参考 `/home/yanjun/ykj/labwam_g1/code/low`，独立部署到本目录，不修改参考工程。
`REFERENCE_SOURCES.json` 记录原文件SHA256；改动包括JAR关节顺序、六项操作、观测初始姿态、独立推理子进程。
参考代码许可证保存在 `REFERENCE_LICENSE.txt`，SONIC归档保留自身许可证。
控制代码和SONIC/ONNX/URDF归档均从同一ModelScope仓库下载并校验。
`.venv-control` 是项目uv环境，**只读继承**原 `.conda_controller` 的Python3.10依赖；没有修改原环境，
因此启动仍依赖原解释器及其site-packages存在。GPU推理继续使用独立 `.venv` Python3.12。

下面每行都是一条完整命令，分别执行，无需手动折行。安装与配置检查不会初始化机器人通信：

```bash
cd /home/yanjun/projects/JAR_G1_Deploy_20261112
bash bundle/deployment/run.sh setup-operator
bash bundle/deployment/run.sh operator --check-config
```

默认启动为只读诊断；可以读状态、计算目标和查询模型，但禁止释放原厂模式或发送电机/夹爪指令：

```bash
bash bundle/deployment/run.sh operator --task DISPENSE_TO_HELD_TUBE
```

以下是后续现场操作入口，**本次没有执行过此命令或控制机器人**。需要交互终端、机器人支撑和独占控制通道。
使用SONIC低层全身站立时，必须退出其它PICO/WAM/arm_sdk/低层控制程序。
GPU推理服务可以继续运行，入口复用同一操作的服务；若操作不同则先按下文stop-policy/start-policy切换。

```bash
bash bundle/deployment/run.sh operator --task DISPENSE_TO_HELD_TUBE --execute
```

| 操作 | 行为 |
| --- | --- |
| 打开程序 | idle，仅观察，不接管；不发送身体或夹爪命令 |
| 第一次回车 | 释放原厂模式，启用SONIC站立，导航始终为零；双臂KP/KD及重力补偿为零，不写夹爪 |
| 等到standing_ready后第二次回车 | 双臂和Dex1缓慢对齐录制的observation初始姿态；臂限速0.25rad/s |
| 等到ready后第三次回车 | 模型、相机和状态就绪后才开始JAR推理，保持零行走指令 |
| d | 立即全身阻尼，清空预测、补偿及旧会话；停止模型客户端，身体tau/KP归零、KD=8 |
| r | 从阻尼/故障/暂停状态重新站立并恢复所选初始姿态；到ready后还需一次回车启动推理 |
| p | 在idle/standing_ready/ready/damped选择操作编号、episode、frame并保存；不是自动切换模型 |
| Ctrl+C | 阻尼并清理本次客户端；不会自动张开夹爪 |

`standing_ready` 表示初始化完成且至少等待2秒，**不是机器人已经稳定站立的检测器**。
`d` 会停止主动平衡，需要现场支撑；`r` 不是倒地起身功能。模型没有动作成功/任务完成检测器，默认持续推理。
可用 `--chunks 100` 限制预测次数，正常结束后保持站立并进入paused；现场按r重新初始化。
机器人观测停止或安全监测失败由原SONIC安全链处理；相机/推理客户端故障停止更新上肢目标并进入fault。
旧epoch、重复序号和超过0.25秒的命令被丢弃，上肢回到实测位置保持。

初始姿态目录包含349段、145173帧的上肢观测，来源仅为 `observation.state[15:31]`。
默认DISPENSE为episode132、frame0，**不会重放录制的腿部或腰部角度**。
`p`菜单输入为三个非负整数，例如如下单独一行，随后回车：

```text
2 132 0
```

重力补偿在 `gravity.py`：根据实测body29关节与pelvis IMU的wxyz四元数，把世界重力旋到机器人基座，
用Pinocchio计算命名关节的广义重力，再仅取双臂14维叠加到SONIC低层body_tau。
使用不带Dex3占位手惯量的G1 29DOF URDF。默认scale1，限幅min(10Nm,URDF effortLimit)，腕pitch/yaw为5Nm，
变化率10Nm/s。首帧、禁用、阻尼、重启及超过0.25秒的计算间隔清零；无效IMU拒绝计算，不假设机器人竖直。
Dex1、转接件和夹持物质量/质心未知，目前没有额外负载补偿。
若有称量结果，可用 `--arm-gravity-payload 文件路径` 传入左右mass_kg/com_m，COM在各自wrist_yaw关节坐标系，
格式示例（数值必须替换为实测值；不建议复制示例直接上机）：

```json
{"left": {"mass_kg": 0.5, "com_m": [0.0, 0.0, 0.1]}}
```

独立ROS domain默认74，仅用于本机控制器与JAR客户端。Unitree反馈仍使用真实机器人DDS域0。
本流程没有训练走路，零导航时只执行Balance ONNX；参考工厂加载Walk文件但本次测试把其执行函数替换为报错并确认零调用。

## 保留的arm_sdk诊断后端

只使用宇树官方 `rt/arm_sdk` 上半身接口，以及 `rt/dex1/{left,right}/{cmd,state}`。
读取 `rt/lowstate`；14个臂关节采用SDK索引15–28，腰部保持接管时的观测位置，不发送腿部关节目标。
此后端要求G1的原厂高层站立控制器正常运行；它不启动或替换底层站立控制，也不切换机器人运动模式。
不能与另一个SONIC/gear_wbc低层写入程序同时控制同一机器人。本包没有假设采集时的31维观测就是电机命令。
profile中的臂限位来自SONIC G1 XML，Dex1行程0–5.4rad；接口增益参考官方示例。

## 命令

先进入工作目录。下面每个代码块内，每行是一条完整命令，分别执行，不需要手动换行。

```bash
cd /home/yanjun/projects/JAR_G1_Deploy_20261112
bash bundle/deployment/run.sh doctor
bash bundle/deployment/run.sh benchmark --all-checkpoints
```

`--all-checkpoints` 需要对应权重全部下载完成。同步中可按已下载操作单独运行以下完整一行命令：

```bash
bash bundle/deployment/run.sh benchmark --task DISPENSE_TO_HELD_TUBE
```
实时传输进度见 `reports/transfer-progress.json`；`reports/checkpoint-benchmark.json` 累积记录已通过的各个step。

启动选定操作的GPU推理服务，默认加载验证最佳权重，只绑定127.0.0.1。可把任务改为导出清单中的其它操作。
这一步没有电机输出，关闭SSH后仍运行。
初始化已预加载完整下载的DISPENSE_TO_HELD_TUBE。切换到其它操作前先停止该服务，并确认对应权重已经下载。

```bash
bash bundle/deployment/run.sh stop-policy
bash bundle/deployment/run.sh start-policy --task TAKE_TUBE --checkpoint best
bash bundle/deployment/run.sh status
```

继续从ModelScope下载后续完成的操作，并逐个运行真实GPU推理验证，直到本轮六项全部到齐且验证完后退出：

```bash
bash bundle/deployment/run.sh start-sync
```

自动验证和真机输出共用项目锁：真机运行时验证会等待；若验证正在占用GPU，真机入口拒绝接管。
需要立即真机操作时，先暂停同步，结束后再继续。以下每行是一条完整命令，分别执行：

```bash
bash bundle/deployment/run.sh stop-sync
bash bundle/deployment/run.sh start-sync
```

连接G1时，先确认它是29DOF、Dex1双夹爪，原厂站立控制运行。相机服务必须提供上述三路图像。
下方相机地址已于2026-11-12连接真实机器人核实，三路相机位于5555端口；若现场IP或端口不同，必须替换。pku4090当前有线网卡为enp132s0。
先运行只读链路；它只订阅状态、相机并查询模型，不建立电机命令发布器。

```bash
bash bundle/deployment/run.sh read-only --interface enp132s0 --camera-endpoint tcp://192.168.123.164:5555 --duration 20
```

旧 `live` 子命令仅为arm_sdk后端，依赖原厂站立，不含上述SONIC站立/初始化/重力补偿。
本次推荐使用前面的operator流程；两种后端不能同时运行。

切换操作前先停止GPU服务，再指定新操作启动；这不会自动启动真机输出：

```bash
bash bundle/deployment/run.sh stop-policy
bash bundle/deployment/run.sh start-policy --task TAKE_DROPPER_AND_ASPIRATE --checkpoint best
```

## 验证结果和边界

实际平台、测量值和已通过项目见 [VALIDATION.md](VALIDATION.md)。

验证记录保存在工作目录 `reports`，包括环境、权重SHA256、真实GPU推理延迟、AR前缀一致性、RPC、限速/失联和隔离DDS测试。
`deployment/fixtures` 是六项操作各一份真实留出观测，来源见其中provenance.json；没有使用action字段作为模型输入或标签。
模拟DDS测试仅在lo与独立DDS domain中运行，不连接机器人。它验证消息和索引，不能代表真机闭环成功率。
已完成训练的离线自回归误差仍高于静止基线；实际抓取、吸液、加液与接触效果需要G1现场验收，不以离线loss替代。

官方接口参考：

- https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/g1/high_level/g1_arm7_sdk_dds_example.py
- https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/robot_control/robot_hand_unitree.py
- https://github.com/unitreerobotics/dex1_1_service

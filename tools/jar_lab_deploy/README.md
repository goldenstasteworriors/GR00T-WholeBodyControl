# pku4090：G1 Dex1 双夹爪 JAR 部署

工作目录为 `/home/yanjun/projects/JAR_G1_Deploy_20261112`。模型、tokenizer、冻结的模型源码及本目录部署代码均从私有 ModelScope 仓库 `ykjykj/JAR-G1-Lab-UpperBody-Observation-20261111` 下载。
`bundle/export_manifest.json` 列出已完成操作和权重SHA256；后续训练完成的操作由同步服务继续下载。
独立uv环境位于 `.venv`，下载用的轻量环境位于 `.venv-download`。不修改旧JAR、Conda环境、CUDA或显卡驱动。

## 输入、输出与运行方式

输入顺序固定为左臂7关节、右臂7关节、左Dex1、右Dex1，角度单位rad。对应原 `observation.state[15:31]`。
相机顺序为ego_view、left_wrist、right_wrist，RGB，240×240；历史为30Hz下 `[t-24,t-16,t-8,t]`，启动时去重。
归一化和英文指令直接读取各操作的 `observation_representation.json`，不读取 action 或初始框。

权重预测未来32帧的观测位置。运行时可以只生成前1帧的16个token；这与完整512-token生成的首帧严格对应，
不是重新训练或改变输出定义。benchmark同时检查1/8/32帧的前缀一致性并记录真实延迟。
推理在独立线程运行，50Hz执行线程对参考位置限速、检查状态时效；旧预测保持上一安全参考，旧机器人状态终止输出。
这是一种慢速参考位置闭环，不能把完整512-token生成速度误称为30Hz控制性能。

## 已配置的机器人接口

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

启动选定操作的GPU推理服务，默认加载验证最佳权重，只绑定127.0.0.1。可把任务改为导出清单中的其它操作。
这一步没有电机输出，关闭SSH后仍运行。

```bash
bash bundle/deployment/run.sh start-policy --task TAKE_TUBE --checkpoint best
bash bundle/deployment/run.sh status
```

继续从ModelScope下载后续完成的操作，直到本轮六项全部到齐后退出：

```bash
bash bundle/deployment/run.sh start-sync
```

连接G1时，先确认它是29DOF、Dex1双夹爪，原厂站立控制运行。相机服务必须提供上述三路图像。
下方相机地址是采集记录中的示例值；若现场IP或端口不同，必须替换。pku4090当前有线网卡为enp132s0。
先运行只读链路；它只订阅状态、相机并查询模型，不建立电机命令发布器。

```bash
bash bundle/deployment/run.sh read-only --interface enp132s0 --camera-endpoint tcp://192.168.123.164:5558 --duration 20
```

现场检查只读结果、状态时效和预测位置之后，才运行下面的真机入口。程序还会要求在交互终端输入 `ARM`。
默认最多20秒；Ctrl+C结束时逐步释放arm_sdk接管权，不自动张开夹爪。

```bash
bash bundle/deployment/run.sh live --interface enp132s0 --camera-endpoint tcp://192.168.123.164:5558 --duration 20
```

切换操作前先停止GPU服务，再指定新操作启动；这不会自动启动真机输出：

```bash
bash bundle/deployment/run.sh stop-policy
bash bundle/deployment/run.sh start-policy --task TAKE_DROPPER_AND_ASPIRATE --checkpoint best
```

## 验证结果和边界

验证记录保存在工作目录 `reports`，包括环境、权重SHA256、真实GPU推理延迟、AR前缀一致性、RPC、限速/失联和隔离DDS测试。
`deployment/fixtures` 是六项操作各一份真实留出观测，来源见其中provenance.json；没有使用action字段作为模型输入或标签。
模拟DDS测试仅在lo与独立DDS domain中运行，不连接机器人。它验证消息和索引，不能代表真机闭环成功率。
已完成训练的离线自回归误差仍高于静止基线；实际抓取、吸液、加液与接触效果需要G1现场验收，不以离线loss替代。

官方接口参考：

- https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/g1/high_level/g1_arm7_sdk_dds_example.py
- https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/robot_control/robot_hand_unitree.py
- https://github.com/unitreerobotics/dex1_1_service

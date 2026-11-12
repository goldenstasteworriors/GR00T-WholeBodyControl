# pku4090 验收记录（2026-11-12）

## 本次初始化与重力补偿增量验证

- 参考工程原文件未修改；部署包和SONIC归档从ModelScope下载，SHA256校验通过。
- 按交付命令运行setup-operator及operator --check-config：真实URDF、Balance/Walk ONNX、349段观测初始姿态可加载，无机器人I/O。
- 重力CPU数值：30组随机合法姿态，势能梯度误差最大4.89e-9Nm，浮动基参考误差4.44e-15Nm。
  14臂以外tau严格为零；限幅、10Nm/s变化率、禁用/超时复位、无效IMU和左侧负载隔离均通过。
- 生命周期模拟采用真实Balance ONNX，DDS/ROS初始化被禁止、运动模式RPC和最终发送器均替换为内存对象。
  三次回车、站立阶段双臂零增益/零补偿且不写夹爪、限速姿态收敛、服务就绪门控、d/r、旧epoch拒绝、
  初始姿态菜单保存和子进程退出处理通过；Walk函数替换为报错后零调用。
- 使用生产BodyCommandSender在内存中验证LowCmd关节到电机力矩映射、CRC、阻尼tau/KP零、KD8及阻尼后的写入锁存。
- 真实机器人只读重力检查：12秒592个body/IMU样本，最大body延迟1.175毫秒，仅计算未下发。
  命令发布器构造和send均设为报错，创建命令发布器及发送尝试为零。
- 真实只读operator idle运行12秒并信号退出，底层Write和ReleaseMode硬拦截；零调用、正常退出。
- 真实三相机、实测机器人观测和DISPENSE/step1000 GPU RPC的客户端集成通过。
  readonly禁止创建ROS目标发布器；模拟execute仅写入内存ROS对象，chunks1确实交付一次预测再发送停止消息。
- 证据：reports/operator-config.json、reports/jar-gravity-numeric.json、reports/jar-operator-mock.json、
  reports/jar-gravity-robot-readonly.json、reports/jar-operator-robot-idle.json、reports/jar-client-integration.json。
- **没有运行operator --execute，没有发送机器人控制命令。实际站立稳定性、初始化轨迹、补偿手感及抓取任务成功率尚未验收。**

本记录对应已从ModelScope下载的 `DISPENSE_TO_HELD_TUBE / step1000` 最佳权重。
SHA256为 `a116ed1b90c26fa7ed1643137fcf5e7d34ceba262d35db58f65703a9bd035f13`。
其它权重仍在同步；后续验收结果以本机reports中的实际记录为准。

## 环境与模型

- RTX 4090，设备可用内存约48GB；驱动580.159.03保持原样。
- 项目独立uv环境：Python 3.12.3，PyTorch 2.8.0+cu128，Transformers 5.12.1，NumPy 2.2.6。
- Unitree SDK 1.0.1使用官方editable源码安装；CycloneDDS Python绑定0.10.2，项目内C库0.10.5。
- 71个预编译依赖包、模型、tokenizer和部署源码从ModelScope下载并校验；未修改旧Conda环境。
- 实际启动并通过了run.sh的install、doctor、benchmark、start-policy、start-sync入口。

真实留出观测上的GPU推理结果，已先热身，时间为单次测量：

| 生成长度 | 延迟 |
| --- | --- |
| 1帧 / 16 token | 0.0916秒 |
| 8帧 / 128 token | 0.6309秒 |
| 32帧 / 512 token | 2.5125秒 |

1帧、8帧的token均与完整32帧输出的对应前缀逐个一致。单份fixture的16维位置MAE约0.0803 rad。
这不是完整验证集成绩或真机操作成功率，也不是30Hz模型推理性能。

## 接口与故障处理

- 真实GPU RPC：返回有限的1×16观测位置；拒绝错误multipart和NaN输入；错误请求后可继续正常推理。
- 相机：三路JPEG字节、base64 JPEG、NumPy RGB均通过；240×240及RGB通道顺序正确。
- 相机历史保持因果时间间隔和视角间时间差；单路时间戳冻结、缺少视角均被拒绝。
- 20秒完整只读CLI回放：独立DDS domain 201、lo网卡、真实三视角留出图像和真实模型。
  记录20个状态样本，其中16个跟踪参考、4个保持参考，覆盖相机冻结后的保持和恢复。
  持续回放RPC延迟中位约0.097秒；全程没有电机命令发布器。
- 限速：100组状态扰动检查通过；臂每20ms最多0.005 rad，夹爪最多0.01 rad。
  过期预测保持，过期状态和超出参考范围的状态偏离终止；不跳变目标追赶。
- 隔离DDS实际消息：臂索引15–28、启动腰部保持、腿部目标不写入、CRC、Dex1左右topic均通过；退出接管权归零。
- 只读构造在禁止创建任何DDS发布器的条件下通过；管道输入ARM不能启用真机入口。

## 本机证据文件

以下均位于部署根目录，路径每行一个：

```text
reports/environment-doctor.json
reports/wheel-download.json
reports/deployment-download.json
reports/checkpoint-benchmark.json
reports/real-policy-rpc.json
reports/camera-protocol.json
reports/guards-and-protocol.json
reports/read-only-safety.json
reports/isolated-dds.json
reports/read-only-integration.json
reports/live-gate.json
reports/transfer-progress.json
reports/modelscope-sync.json
```

同步未完成时最后一份modelscope-sync报告可能尚未生成。同步服务会逐个校验并运行GPU验收，累计记录每个step。
DDS日志和编译缓存位于项目runtime内。

没有连接真实G1执行动作，尚未验收抓取、夹持力、吸液、加液、接触和站立稳定性。
训练的既有完整离线评估仍有误差高于保持状态基线的情况；不能用部署链路通过替代真机任务评估。

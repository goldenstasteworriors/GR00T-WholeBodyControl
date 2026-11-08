# ThinkPad 有线断开后的机器人端平衡

此部署由机器人进程独占身体 LowCmd 发布。ThinkPad 仍运行现有遥操作、
IK、Balance/Walk 和采集；正常时机器人原样转发其电机目标和增益。
机器人接收端使用本地 DDS，断开 ThinkPad 的有线连接不切断该本地反馈。

指令流超过 100 ms 没有新包，或 TCP 连接关闭，即锁定本次进程。
这也会覆盖电脑控制循环超过 100 ms 的停顿，不只是物理拔线。

- 已进入 running：停止行走，保留高度和躯干参考；使用同一个 Balance
  ONNX、原观测缩放、6 帧历史和原增益，在机器人本地以 50 Hz 持续推理。
  检测断线的间隔内也持续更新本地状态历史，避免用过时反馈直接接入平衡。
  腿和腰三关节继续闭环，双臂保持最后实际下发的目标，速度和前馈力矩清零。
- 尚在初始化/ready：冻结现有位置目标，不自动开启站立。
- 已急停：保持阻尼，不自动恢复支撑。
- 重新插线、PICO 四键、键盘初始化均不能解锁；停止两个控制进程后重新启动。
- 正常退出、急停仍发阻尼。断线时 ThinkPad 的急停无法保证送达机器人，
  需要机器人侧可用的急停手段；重新插线后的原 TCP 会话若仍存在，只允许阻尼/退出。

冻结目标或运行 Balance 都无法保证任何姿态下绝不倒地。读模式和离线测试
只能验证数据链路/状态转换/推理，不能替代有支撑条件下的实机验收。

## 停止断线锁定的进程

下面每个代码块各执行一行。停止机器人进程会进入阻尼，先确保有可靠支撑。

ThinkPad 上停止机器人进程：

```bash
ssh -- g1 'cd /home/unitree/GR00T-WholeBodyControl/.worktrees/onboard-body-hold && bash scripts/onboard_body_service.sh stop'
```

退出原采集会话后，仍使用原来的三个参数一键命令重新启动。
机器人进程没有自动重启或自动重连。

```bash
bash scripts/collect_on_thinkpad.sh default '任务描述' '项目名'
```

## 部署环境

机器人代码使用独立 Git worktree，不覆盖既有相机修改。
Python 环境是该 worktree 下 `.venv_body`，使用已有 Python 3.10。
依赖只有 numpy、PyYAML、msgpack、onnxruntime、官方 unitree_sdk2py；
CycloneDDS 使用机器人已有安装，不更改系统 Python、CUDA 或驱动。
安装脚本为 `install_scripts/install_onboard_body_env.sh`。机器人不能联网时，
从 ThinkPad 转存 ARM64 wheels 到 `.runtime/onboard_body_wheels` 后离线安装。
官方 SDK 使用源码 editable 安装，保留其 ARM64 CRC 库；Balance 模型和 CRC
是 Git LFS 资产，机器人未安装 Git LFS 时需要单独传送并核对 SHA256。

新增的 TCP 转发会增加通信耗时；不能把机器人本地推理速度等同于完整遥操延迟。

## 验收边界

部署后正在运行的旧采集进程不会自动切换到新链路。
必须先安全退出旧采集，再启动新采集。完整 tmux 启动和有支撑断线站立
尚未实测时，不应把该版本称为已完成实机验收。

已完成的只读验证：原策略与本地实现 200 组输入对比，观测历史逐项一致、
输出角度差为 0；ThinkPad 至机器人连续发送 100 包，停发后进入 Balance
锁定，拒绝新连接/新位置指令，阻尼及退出路径确认成功。只读接收端不创建
LowCmd 发布器，上述验证没有发送电机指令。

# ThinkPad 双 Dex1 采集

工作区：`/home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard`。
新机器人 DDS 地址为 `192.168.123.164`；ThinkPad 的有线网卡为
`enx00e04c36c4e8`，地址为 `192.168.123.222`。不要继续使用旧的
`enxdc045a1d933b`，也不要启动 Inspire Modbus bridge。

## 启动检查

下面命令按原样逐行运行；反斜杠表示命令跨行续写。

```bash
cd /home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard
conda activate decoupled_vla_collection
source "$CONDA_PREFIX/setup.bash"
python gear_sonic/scripts/check_dex1_connection.py --network enx00e04c36c4e8
```

检查程序只订阅状态，不创建命令发布器。左右各需持续收到一个电机状态，
身体需收到 LowState；零角度也是有效反馈。机器人端应运行官方
`dex1_1_gripper_server --network eth0`。服务安装在
`/home/unitree/dex1_1_service`，依赖只放在其 `.deps` 内。

## 采集入口

先执行同参数预检，不会启动控制或写入数据集：

```bash
python gear_sonic/scripts/launch_decoupled_vla_collection.py \
  --interface enx00e04c36c4e8 \
  --hand-type dex1 \
  --camera-host 192.168.123.164 \
  --task-prompt "grab the red bottle" \
  --dataset-name pour_water_between_beakers_dex1_20261014 \
  --hand-task grab_middle_beaker \
  --no-pico-data-streamer \
  --preflight-only
```

预检通过后去掉 `--preflight-only` 才会启动真实控制。Dex1 模式启动前
自动检查身体/双夹爪 DDS 和相机 TCP 端口。TCP 可达不代表相机图像已经
验证。预检失败不会杀掉已有 tmux 会话、释放机器人运动模式或覆盖数据集。
不要把旧 Inspire 数据集与新的 Dex1 数据集混用；默认不覆盖已有数据。

`--pico-vis-vr3pt` 可按原流程额外添加。`--startup-t-pose` 会影响真实机器人的
启动姿态，不是连接测试选项。

## 手部与数据语义

- PICO 原有开合逻辑：未按下为开、按下为抓。`grab_middle_beaker` 保留为任务名；
  Dex1 不使用 Inspire 的六指姿态 JSON。
- 官方控制参考：角度单位 rad，闭合 0、打开 5.4，kp=5、kd=0.05。
  每次目标限制在反馈位置 ±0.18 rad 内。位置超出标定范围会报错，
  不自动标定或更改零点。
- `observation.state`、`action.wbc`：31 维，顺序为身体29、左夹爪1、右夹爪1。
  `action.wbc` 保存实际下发的限幅目标，不从手势重算。
- `teleop.left_hand_joints`、`teleop.right_hand_joints` 各 1 维。
  元数据标记 `g1_dex1_31dof` 和 rad，不冒充 Inspire/旧43维模型兼容数据。
- 原七关节手部表示仅用于现有 WBC/手腕 FK 的结构占位，状态保持零值。
  原手部几何不是 Dex1 几何；不声称支持 Dex1 接触仿真或指尖位置计算。
- `--hand-type inspire` 保留旧采集格式。

官方参考：
[Dex1 服务](https://github.com/unitreerobotics/dex1_1_service)、
[遥操作控制](https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/robot_control/robot_hand_unitree.py)。

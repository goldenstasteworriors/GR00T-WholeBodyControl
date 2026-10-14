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
自动检查身体/双夹爪 DDS，并要求收到可解码、捕获时间戳持续更新的相机 RGB 帧。
预检失败不会杀掉已有 tmux 会话、释放机器人运动模式或覆盖数据集。
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

## 2026-10-14 配置与验收记录

- 新机器人服务：`sonic-dex1.service`，已启动并启用开机自启；
  系统服务文件为 `/etc/systemd/system/sonic-dex1.service`。
- 右侧 ID 0 → `/dev/ttyUSB1`；左侧 ID 1 → `/dev/ttyUSB2`。
- ThinkPad 上 8 秒只读订阅：LowState 8348 条、左夹爪4001条、右夹爪4004条；
  夹爪反馈位置约左0.9022 rad、右0.7203 rad。均为实时硬件反馈。
- 离线测试已覆盖：缺失/过期反馈拒绝、0.18 rad 限幅、命令与原生状态的31维导出、
  旧 Inspire 41维格式保留、ThinkPad 实际环境的导入与 CLI 参数解析。
- 一键启动预检：机器人及双夹爪通过，`192.168.123.164:5555` 拒绝连接。
  新机器人的相机 ZMQ 服务尚未部署，尚未验证图像、PICO 遥操作或完整录制。
  没有启动身体控制、开合测试、T-pose 或覆盖旧数据集。
- 未重启机器人验证自启；已检查 systemd 的 active/enabled 状态。

机器人端官方服务基线为 `ced3216290bb22dae356213993786beeaa7143ab`，
项目内 SDK 基线为 `63096d0ac0c5d2dec9d6e0c22cd5233410ca2f36`。
机器人无 GitHub DNS 解析，使用本机通过 SSH 获取的官方仓库 Git bundle 部署。
机器人 Git 提交为 `d221888`，包括零力矩探测和串口失败不发布旧状态的修正；
补丁和服务文件备份在本采集仓库 `install_scripts/dex1/`。
官方仓库不具备推送授权，修正通过采集仓库备份。

libfmt-dev 6.1.2+ds-2、libserialport-dev/libserialport0 0.1.1-3 均解包在
服务 `.deps` 内，不安装到系统。下载包校验与机器人 Ubuntu 包索引相符。
SDK 使用 `-DBUILD_EXAMPLES=OFF`、`-DCMAKE_INSTALL_PREFIX=<服务目录>/.deps`
配置安装；服务编译时通过 `-I`/`-L` 指向该前缀，临时编译目录也位于 `.deps/tmp`。
systemd 服务中的 `LD_LIBRARY_PATH` 保证串口动态库可被找到。

在 ThinkPad 登录机器人后可以查看或重启本服务（每行一条命令）：

```bash
ssh g1
systemctl status sonic-dex1.service --no-pager
sudo systemctl restart sonic-dex1.service
```

登录时若询问 ROS 版本，输入 `1`。无需修改 CUDA、显卡驱动、系统 Python 或 ROS 安装。

## 机载 D435i RGB 配置（2026-10-14 后续检查）

相机直接连接机器人。现有 USB/UVC 驱动已通过本机实测：90/90 帧有效，
640×480 RGB，29.996 FPS。固定的 USB 视频接口 ID 为 `254843062005`；
使用 `/dev/v4l/by-id/` 路径，避免重启后 `/dev/video8` 编号改变。
采集使用 `ego_view` 彩色图像，不包含深度、红外或 IMU。

机器人相机工作区及虚拟环境（各行一个路径）：

```text
/home/unitree/GR00T-WholeBodyControl
/home/unitree/GR00T-WholeBodyControl/.venv_camera
```

独立环境使用 uv 管理的 Python 3.10.22；uv、Python、缓存和编译临时文件均在
项目 `.runtime/` 内。RGB 服务依赖列在 `install_scripts/requirements_camera_d435i.txt`，
不依赖 PyTorch、CUDA、OAK SDK 或 RealSense SDK。服务模板为
`systemd/sonic_d435i_camera.service`，使用 `unitree` 用户和 ZMQ 端口5555。

**部署状态：** 生成最终 UVC 服务配置时，ThinkPad 的 `10.90.234.236` SSH
突然超时且 ping 不通。因此最终配置尚未完成远端同步确认，尚未安装/启用相机
systemd 服务，也尚未验证 ThinkPad 收图和完整启动预检。恢复连接后需要先检查
是否已有前台相机进程，再完成服务安装和验收；不能仅凭本机出图宣称采集链路通过。

后续机器人端安装命令如下，每行一条命令；必须先结束本次前台相机测试，
避免同时占用相机或5555端口：

```bash
cd /home/unitree/GR00T-WholeBodyControl
sudo install -m 0644 systemd/sonic_d435i_camera.service /etc/systemd/system/sonic_d435i_camera.service
sudo systemctl daemon-reload
sudo systemctl enable --now sonic_d435i_camera.service
systemctl status sonic_d435i_camera.service --no-pager
```

技术记录：官方 PyPI ARM wheel 与机器人 Ubuntu 20.04 的系统库不兼容。
已按[官方源码构建说明](https://github.com/realsenseai/librealsense/blob/v2.55.1/wrappers/python/readme.md)
编译 SDK v2.55.1（源码 `e196cefa896e312d79c2df400c7623aa1e9c62ac`），
本地 wheel 可导入，但 RSUSB 模式捕获持续超时，因此最终 RGB 服务采用已验证的 UVC 驱动。
`install_scripts/build_d435i_sdk.sh` 仅保留为可选构建记录，不是当前 RGB 服务的安装前提。
SDK 返回设备序列号 `401622070741`，它与 USB 视频接口 ID 不同。
没有改动固件、内核、系统库、CUDA 或显卡驱动。

ThinkPad 端额外验证：冻结捕获时间戳的模拟图像流被预检正确拒绝；
SDK 编译期间3秒 DDS 复检收到身体3130条、左夹爪1501条、右夹爪1500条，全部有效。
相机与采集端的系统时钟存在约数分钟差异，当前预检只检查时间戳持续前进；
跨机器时间戳之差不能直接当作传输延迟。尚未修改系统时钟或验证严格时间同步。

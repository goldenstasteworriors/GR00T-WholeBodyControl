# ThinkPad 双 Dex1 采集

**2026-10-20 当前任务：暂停有线与机载部署完善，优先验证 ThinkPad 无线 PICO 采集，数据保存在 ThinkPad。
请按 [ThinkPad 无线 PICO 采集流程](thinkpad_wireless_collection.md) 操作，其中包含显式启动 PC Service 的命令。
下文保留历史记录。**

**2026-10-19 历史拓扑复核：当时用户要求 PICO 直连机器人，恢复机器人端 DHCP/PICO 服务。
下文“ThinkPad PICO”章节是此前误选的部署方案，不再作为当前启动指引。
本次启动的 ThinkPad DHCP 和 PC Service 已停止，夹爪网页保留。
最新阻塞项与验证范围见文末“机载 PICO 拓扑复核”。不要直接照搬旧机器人路径。**

多场景夹爪配置、PICO 有线/无线服务和最新预检见本文末尾。

当前入口默认录制 D435i 主视角和左右腕三路 RGB；最新配置、同步限制与直接编辑
夹爪 JSON 的方法见本文末尾的“三相机同步采集”章节。下方单路相机测量为历史记录。

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
export PYTHONPATH="$PWD"
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

**2026-10-19 部署状态：** 已恢复 ThinkPad 连接并完成代码同步。
机器人相机服务已安装，状态为 active/enabled，重启计数0；ThinkPad 连续接收
241帧不同图像，640×480 RGB，实测30.012 FPS。
身体、双 Dex1、真实相机图像的一键启动预检均通过。
尚未执行真实身体控制、PICO 操作录制和夹爪动作验收；需要现场确认机器人放置与夹爪周围状态。

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

## Dex1 开口测试与扳机角度（2026-10-19）

角度单位 rad，官方控制范围为0（闭合）到5.4（打开），不是开口宽度毫米。
左右手可以分别配置；实物合适的角度应先用操作工具测定。
PICO 扳机值大于0.5视为按下，其余视为松开，保留原来的二值扳机逻辑。

永久操作工具为 `gear_sonic/scripts/exercise_dex1_grippers.py`，按用户要求保留，
不是验证后应删除的临时测试脚本。以下命令每行一条，均在上述 ThinkPad 工作区和采集环境内运行。

只读查看两只夹爪角度，不创建命令发布器：

```bash
python -m gear_sonic.scripts.exercise_dex1_grippers --network enx00e04c36c4e8 --duration 5
```

实际动作示例（现场清空夹爪周围，并先退出采集控制进程）：

```bash
python -m gear_sonic.scripts.exercise_dex1_grippers --network enx00e04c36c4e8 --left-q 1.2 --right-q 1.0 --speed 0.3 --execute --return-to-start
```

`--left-q`、`--right-q` 指定本次测试的目标角度；省略某侧即不控制该侧。
`--speed` 是目标斜坡速度，默认0.5 rad/s，最大1 rad/s；`--duration` 是每段到达目标的超时，默认15秒。
`--return-to-start` 会在到达后返回初始角度。超时、过期反馈、Ctrl+C或正常结束都会撤掉位置增益。
保留每次目标相对反馈的±0.18 rad限制。当前项目内的夹爪操作与采集控制共用进程锁，
避免同时控制同一只夹爪；其它计算机或不使用本项目的发布器不受这个文件锁约束。

采集时这两个参数各接收两个数，顺序均为**左、右**：

- `--dex1-released-q`：松开扳机的目标，默认5.4、5.4。
- `--dex1-pressed-q`：按下扳机的目标，默认0、0。

下面为自定义角度的完整预检示例；反斜杠表示跨行续写。示例角度用于说明参数，
不代表已经对当前容器完成抓取标定。

```bash
python gear_sonic/scripts/launch_decoupled_vla_collection.py \
  --interface enx00e04c36c4e8 \
  --hand-type dex1 \
  --dex1-released-q 4.5 4.6 \
  --dex1-pressed-q 0.8 0.9 \
  --camera-host 192.168.123.164 \
  --task-prompt "grab the red bottle" \
  --dataset-name pour_water_between_beakers_dex1_20261019 \
  --hand-task grab_middle_beaker \
  --no-pico-data-streamer \
  --preflight-only
```

确认实际抓取角度和现场条件后去掉最后一行的 `--preflight-only`，并去掉上一行末尾的反斜杠，
即可启动采集。角度沿 launcher → control CLI → WBC 配置 → 各侧硬件对象传递，
并随控制配置保存到数据集元数据；数据集仍记录真实状态和实际限幅命令，31维格式保持一致。
不需要修改 Inspire 的六指姿态 JSON。

本次验证：双夹爪工具只读运行通过；5秒身体5219条、左2503条、右2501条反馈，全部有效；
上述自定义角度完整预检通过；三个实际采集组件的导入、完整生成参数和 CLI 解析通过；
无硬件命令的模拟验证覆盖自定义角度映射、0.18 rad限幅、无效目标/过期反馈拒绝、
同侧命令互斥和退出撤增益。尚未执行夹爪真实动作及完整 tmux 采集录制验收。

## 网页调节夹爪并保存采集配置

网页后台运行在连接机器人网线的 ThinkPad 上；可以使用 ThinkPad 本机浏览器，
也可以从当前工作站通过 SSH 转发访问。后台仅监听本机回环地址，不需要安装新的 Web 依赖。
本次已在 ThinkPad 的 `dex1_gripper_web` tmux 会话启动，默认只读。

浏览器地址：

```text
http://127.0.0.1:8766
```

如果使用当前工作站而不是 ThinkPad 浏览器，先建立下面的 SSH 转发。
这是一个持续运行的命令，保留终端；本次已经建立转发时无需重复执行：

```bash
ssh -N -L 127.0.0.1:8766:127.0.0.1:8766 -- thinkpad
```

今后需要重新启动网页后台时，在 ThinkPad 逐行运行下面命令。
如果现有网页服务仍在运行，可以直接使用，不要重复启动同一端口。

```bash
cd /home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard
conda activate decoupled_vla_collection
source "$CONDA_PREFIX/setup.bash"
export PYTHONPATH="$PWD"
python -m gear_sonic.scripts.run_dex1_gripper_web --network enx00e04c36c4e8 --port 8766
```

操作顺序：

1. 确认实际反馈在线，退出正在运行的采集控制。清空夹爪周围后，点击“启用夹爪控制”。
   启用时以实际角度为目标，不跳到默认开口。
2. 分别拖动左右滑块或填写目标角度，点击该侧“应用目标”。默认以0.3 rad/s缓慢移动。
   拖动滑块或改数字本身不会发送动作。页面的大数字是实际反馈，调节框是目标值。
3. 找到合适的松开位置后，在表格“松开扳机”一行，分别点击两侧“记下实际位置”；
   找到合适的夹持位置后，在“按下扳机”一行同样记录。四个角度也可以直接编辑。
4. 用“试松开扳机位置”和“试按下扳机位置”预览两种状态，再点击“保存采集配置”。
   保存不产生运动。“下载 JSON”可另外导出一份副本。
5. 把夹持物放稳后点击“停止控制”，再启动采集。停止、网页关闭/切到后台或心跳中断时，
   后台会撤掉位置保持增益；夹爪不再主动保持夹持。

配置默认保存在 ThinkPad 项目磁盘上，重启后保留，不是系统临时目录：

```text
/home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard/.runtime/dex1_gripper_profile.json
```

网页“复制参数”会提供该文件的完整路径。配置包含 version、unit、released_q、pressed_q；
两个角度数组均按左、右排序。`--dex1-profile` 会覆盖命令行中的两组直接角度参数，
建议只选一种方式。配置在采集启动时读取；之后网页保存的修改在下次启动采集时生效，
不会热更新正在运行的机器人控制。

完整采集预检命令如下，反斜杠表示跨行续写。先在网页保存自己的配置，再运行：

```bash
python gear_sonic/scripts/launch_decoupled_vla_collection.py \
  --interface enx00e04c36c4e8 \
  --hand-type dex1 \
  --dex1-profile .runtime/dex1_gripper_profile.json \
  --camera-host 192.168.123.164 \
  --task-prompt "grab the red bottle" \
  --dataset-name pour_water_between_beakers_dex1_20261019 \
  --hand-task grab_middle_beaker \
  --no-pico-data-streamer \
  --preflight-only
```

预检通过、现场准备好后，去掉最后一行和上一行末尾的反斜杠，即为正式采集命令。
若网页仍持有夹爪控制，采集会在启动身体控制前拒绝启动；点击网页“停止控制”后再试。
网页可保持打开用于只读观察。

2026-10-19 验证：真实网页收到左右约0.90/0.72 rad反馈，默认只读；模拟夹爪完成网页启用、
左右独立调节、读取实际位置、两种扳机状态预览、保存配置和停止流程；无效输入被拒绝且错误持续可见；
后台覆盖速度斜坡、控制权、反馈过期和1秒心跳超时停止。已检查桌面及390像素宽布局。
保存的测试配置实际经过身体/双夹爪/相机预检，并验证占用夹爪时采集提前拒绝启动。
验证过程没有对真实机器人发送夹爪或身体动作，也没有保存示例值为用户的正式配置。

## 三相机同步采集与直接填写夹爪 JSON（2026-10-19）

可以直接在 ThinkPad 编辑配置，不必经过网页。下面每行一条命令：

```bash
cd /home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard
mkdir -p .runtime
nano .runtime/dex1_gripper_profile.json
```

文件内容示例（整体复制；数值请替换为你测试得到的角度）：

```json
{
  "version": 1,
  "unit": "rad",
  "released_q": [4.2, 4.4],
  "pressed_q": [0.8, 0.9]
}
```

每个数组第一项是左夹爪，第二项是右夹爪。released_q 对应松开扳机，
pressed_q 对应按下扳机；允许范围为0至5.4 rad，数值越大越打开。
JSON 不支持注释，最后一项后不能加逗号。nano 中按 Ctrl+O、回车保存，Ctrl+X 退出。
文件在启动时读取；修改后需重启采集才能生效。上面的四个角度只是格式示例，
没有写入用户的正式配置，也不代表实物抓取标定结果。

三路相机全部连接机器人，已通过遮挡左腕镜头确认左右：

| 数据视角 | 设备 | 固定 USB 视频接口 |
| --- | --- | --- |
| ego_view | D435i 彩色 | 254843062005 / video-index0 |
| left_wrist | JR0001 | JR0001 / video-index0 |
| right_wrist | JR0002 | JR0002 / video-index0 |

机器人上的 sonic_d435i_camera.service 已更新为三路服务，端口仍为5555。
服务按机器人同一主机时钟将帧分组，组内时间差不得超过30ms；
采集端必须收到完整、持续更新的三路 RGB 组，才能写入数据。
缺一路、格式错误或接收后超过250ms没有新组时暂停添加帧，恢复后继续。
冻结时间戳不会刷新有效期。服务器也丢弃超过250ms的源帧。

同步时间戳是 USB 图像在主机读取完成的时间，**不是传感器曝光时间**。
这是软件时间同步，不是硬件触发、几何标定或严格的相机与身体状态曝光同步。
机器人与 ThinkPad 时钟存在偏差，因此接收新鲜度使用 ThinkPad 的单调时钟，
不直接相减两台机器的墙上时钟。

完整三路预检命令如下；反斜杠表示跨行续写。先保存自己的夹爪配置：

```bash
python gear_sonic/scripts/launch_decoupled_vla_collection.py \
  --interface enx00e04c36c4e8 \
  --hand-type dex1 \
  --dex1-profile .runtime/dex1_gripper_profile.json \
  --camera-host 192.168.123.164 \
  --record-wrist-cameras \
  --camera-sync-tolerance-ms 30 \
  --camera-max-hold-ms 250 \
  --data-exporter-frequency 30 \
  --task-prompt "grab the red bottle" \
  --dataset-name pour_water_between_beakers_dex1_three_camera \
  --hand-task grab_middle_beaker \
  --no-pico-data-streamer \
  --preflight-only
```

预检通过后，删除最后一行并去掉上一行末尾的反斜杠，即可正式采集。
网页须先停止夹爪控制。默认不覆盖已有数据；新任务使用新的 dataset-name。
record-wrist-cameras 已默认开启，显式写出便于检查；兼容旧单路采集时才使用
--no-record-wrist-cameras。

三路预览命令（单行运行，不发送机器人动作）：

```bash
python -m gear_sonic.scripts.run_camera_viewer --camera-host 192.168.123.164 --camera-port 5555
```

每个样本保存 observation.images.ego_view、observation.images.left_wrist、
observation.images.right_wrist 三张 RGB，分别编码为三条视频；不拼接成单张图片。
此外保存 observation.camera_timestamps（三路顺序为主视角、左腕、右腕）、
observation.camera_sync_skew_ms 和 observation.camera_receive_age_ms，方便检查同步质量。
身体与 Dex1 的 state/action 仍是31维。源相机配置30Hz，采样30Hz时允许短暂复用
一整组三图，不会将不同同步组的左右图拼到一起；连续视频帧不保证都是独立曝光。

验收：三路真实图像各30帧写入同一 episode，三条 MP4 均可解码为30帧640×480；
Parquet 为30行，记录的最大组内时间差29.451ms。只读身体、双 Dex1、三路相机的
启动器预检通过。修复同步器一次取出多组却只发布最后一组的问题后，20秒实测
375个不同同步组，约18.74组/秒，最大组内时间差29.761ms，无无效组。
**尚未达到稳定30个独立同步组/秒**；采样频率不应被理解为独立相机组频率。
同步器与采集端验证覆盖缺路、陈旧/冻结时间戳、异常帧、组内时间差和断流后的结束保存。
没有启动身体控制、真实夹爪动作或 PICO 正式录制。

## 多场景夹爪配置与 ThinkPad PICO（2026-10-19）

网页新增“已存场景”和“保存名称”。例如保存为 red_bottle、middle_beaker、
small_cup；每组独立保存松开/按下扳机的左右四个角度。换名称可另存，加载已有
场景后可以更新；加载和保存均不发送动作，试位置仍需启用控制并单独点击。
切换场景不会热更新已运行的采集进程。旧 --dex1-profile 文件方式继续兼容，
不能与 --dex1-preset 同时使用。

命名配置目录：

```text
/home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard/.runtime/dex1_profiles
```

当前用户已有配置已复制为 default.json：松开[5.4,5.4]，按下[2.5,2.5]。
原 dex1_gripper_profile.json 保留。其它场景需按实测另存，没有预设未经标定的角度。
各 JSON 仍采用 version=1、unit=rad、released_q、pressed_q 格式；
也可直接编辑目录中的不同 JSON 文件。场景名只允许字母、数字、下划线和连字符。

采集选场景示例（单行参数，不与 --dex1-profile 同用）：

```bash
--hand-type dex1 --dex1-preset default
```

### PICO 的服务与连接位置

本流程在 ThinkPad 上运行 PC Service 和采集，而不是在机器人上运行 PICO 服务。
PicoStreamer 原本会自动启动或复用本机 /opt/apps/roboticsservice/RoboticsServiceProcess；
--no-pico-data-streamer 只关闭可选的附加姿态数据进程，不关闭遥操作或扳机输入。
本轮增加独立服务命令和只读头显检查，并在真实 PICO 采集前默认要求检查通过，
避免仅机器人/相机在线就启动身体控制。--preflight-only 现在也检查 PICO；
--no-pico-preflight 仅供不启动控制的独立诊断使用。

[官方连接说明](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client)
要求头显与 PC 网络互通、启动 PC Service，在 XRoboToolkit 中连接并开启 Send。
Wi-Fi 和有线以太网都走这一网络链路；普通 USB 充电/数据线不属于本流程的有线方式。

推荐有线拓扑：

```text
PICO USB-C → USB转网口 → 同网段交换机/机器人同网段网口
ThinkPad 有线网口 → 同一网络（192.168.123.222）
机器人 → 同一网络（192.168.123.164）
PICO 应用 PC Service → 192.168.123.222
```

接机器人网口时必须确认该端口与上述网段实际互通，不是所有接口都默认互通。
不要给 PICO 填192.168.123.164；那是机载 PICO 服务流程，与本次 ThinkPad 采集不同。

先在 ThinkPad 项目目录运行，每行一条命令：

```bash
bash scripts/thinkpad_pico_services.sh start-wired
bash scripts/thinkpad_pico_services.sh status
```

start-wired 需要 ThinkPad sudo 密码。它不修改网卡地址、路由、NetworkManager、
系统配置文件，只运行项目管理的 DHCP 和 PC Service。DHCP 仅向原 PICO 转网口
MAC dc:04:5a:1d:93:3b 分配192.168.123.200，不提供默认网关或DNS。
更换转接器时必须按实际MAC设置 PICO_ETH_MAC，不应凭型号猜测。

如果使用 Wi-Fi，只启动服务，不需要上述项目 DHCP：

```bash
bash scripts/thinkpad_pico_services.sh start
```

头显与 ThinkPad 连同一可互通 Wi-Fi，PICO 的 PC Service 地址填 ThinkPad 的
Wi-Fi IPv4（本次读取为10.7.39.74，可能变化，以 status 为准）。
校园网可能隔离客户端，同一SSID不能保证可互通。

PICO 应用连接后显示 WORKING，开启头显、左右手柄 tracking 和 Send。
只读验证命令（不启动机器人控制）：

```bash
python -m gear_sonic.scripts.check_pico_connection --timeout 10
```

出现 PICO OK 才说明收到持续更新的头显和双手柄姿态及有效扳机值。
进程存在不等于头显已连接。不要同时连接机载服务和 ThinkPad 服务。

服务停止命令，只在结束采集后执行：

```bash
bash scripts/thinkpad_pico_services.sh stop
```

### 多场景三相机采集

下面反斜杠表示跨行续写；先保存自己的场景，再选对应名称：

```bash
python gear_sonic/scripts/launch_decoupled_vla_collection.py \
  --interface enx00e04c36c4e8 \
  --hand-type dex1 \
  --dex1-preset default \
  --camera-host 192.168.123.164 \
  --record-wrist-cameras \
  --camera-sync-tolerance-ms 30 \
  --camera-max-hold-ms 250 \
  --data-exporter-frequency 30 \
  --task-prompt "grab the red bottle" \
  --dataset-name pour_water_between_beakers_dex1_three_camera \
  --hand-task grab_middle_beaker \
  --no-pico-data-streamer \
  --preflight-only
```

预检通过后删除最后一行和上一行末尾反斜杠，才会启动真实控制。
场景名称及解析后的 JSON 路径保存进 exporter 元数据，控制元数据保留实际四个角度。

本轮验收：多场景保存、切换、更新、禁止误覆盖其它场景、名称路径校验及无动作通过；
实际前端 JavaScript 通过模拟DOM与HTTP后端测试（未完成浏览器布局验收）。
真实 ThinkPad 的 default 场景加载、身体/双Dex1/三相机只读预检通过。
PC Service 启动、重复启动复用和 SDK 连接本机60061成功；
当前未收到头显数据，检查正确返回未就绪。DHCP 参数语法检查通过，
已用 ThinkPad sudo 验证启动：仅监听指定有线网卡、只提供静态租约、DNS关闭。
用户选择稍后连接实物，尚未完成头显有线链路验收。
没有启动身体控制或真实夹爪动作。

## 机载 PICO 拓扑复核（2026-10-19）

已重新阅读 data_collection.md、decoupled_vla_collection.md、vr_teleop_setup.md、
onboard_pico_services.sh、机载 wrapper、当前 launcher/teleop/exporter、Dex1 与相机实现，
并对照旧机载流程提交681ba4c、4145765及实际两台机器状态。

| 项目 | 核查结果 |
| --- | --- |
| 旧部署拓扑 | 机器人 eth0 上 DHCP + ARM64 PC Service；PICO 指向192.168.123.164；旧机载 wrapper 在机器人采集并保存 |
| 当前用户接线 | PICO USB转网口直连机器人；不应擅自改成 ThinkPad 作为服务端 |
| 新机器人目录 | /home/unitree/data_collection/GR00T-WholeBodyControl 不存在；现有项目是 /home/unitree/GR00T-WholeBodyControl |
| 新机器人部署 | 只有相机环境、相机服务和 Dex1 服务；无旧采集 Conda、ARM64 PICO runtime及其 Qt 环境 |
| SDK 地址 | 项目绑定 init() 无远端参数；实际连接127.0.0.1:60061。只将头显改连机器人，ThinkPad SDK 不会自动读取机器人服务 |
| 错误自动启动 | 上一轮预检强制启动 ThinkPad 服务，现已移除；预检仅检查数据，不选择服务部署位置 |
| 重复 DHCP | 此前在 ThinkPad 新建的 DHCP/PC Service 已停止；没有关闭机器人现有热点服务 |
| 旧脚本端口检测 | 机器人热点 dnsmasq 监听10.42.0.1，旧脚本检查任意UDP67会误拒绝有线DHCP；恢复时须限定实际绑定地址冲突 |
| Dex1 场景 | default 保留用户的松开[5.4,5.4]、按下[2.5,2.5]；原JSON未删，命名场景正常 |
| Dex1 输入链路 | 实际 PICO 指尖编码→手部IK→Dex1目标，无动作测试左右两侧及0.5阈值通过 |
| 三路视角 | D435i主视角、JR0001左腕、JR0002右腕，左右通过实物遮挡确认 |
| 相机同步 | 软件读取完成时间分组，最大30ms，不是曝光硬同步；实际独立组率尚未稳定30Hz |
| 数据写盘 | 补测连续两条三视频episode保存通过；原生31维手部格式、场景名路径和四个实际角度元数据传递通过 |
| 尚未验收 | 新机器人PICO实物有线链路、真实身体/PICO遥操作、真实夹爪开合及正式机器人episode |

计算位置需明确后再恢复：若完全恢复旧机载方式，需在新机器人部署完整采集环境；
若采集继续在 ThinkPad，需显式解决机器人 PC Service 与 ThinkPad SDK 的数据通路，
不能仅修改IP或启动DHCP就声称已经打通。未自行安装 Conda、系统Qt或更改驱动。

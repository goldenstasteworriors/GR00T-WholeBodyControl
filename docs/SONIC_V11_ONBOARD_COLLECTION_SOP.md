# SONIC v1.1 机载推理与 Inspire 数据采集 SOP

**按第0～10步依次操作，可以参考sonic官方文档说明。** 首次安装完成第0～8步；以后每次从第9步开始。sonic官方资料放在文末。

部署分工：机器人运行SONIC v1.1推理和RealSense相机；PC运行PICO、Inspire手桥接和数据导出。数据沿用本项目41维格式（身体29＋左手6＋右手6）。

代码块逐行执行，某行失败先处理该错误，再继续；行末 `\` 表示命令在下一行继续。以下步骤用于新目录，已有环境不要重复安装。

“机器人终端”是机器人本机终端或登录到机器人后的SSH终端；“PC终端”是电脑自己的终端。每次新开终端后先执行该步骤里的 `cd`。第0步的临时查询变量不要求跨终端保留，查到的值请写入参数表。只复制代码块，不复制标题、编号和说明文字。

## 0. 确认设备和填写参数

准备：G1＋机载Jetson、Inspire双手、RealSense、Ubuntu x86_64 PC、PICO 4/4 Pro、双手柄、双脚踝追踪器；双方需要联网下载依赖及本项目Git仓库读取权限。

先记录下表。下文命令使用“示例值”；不同的地方在运行前全部替换，左右手IP必须按物理侧别核实。

| 项目 | 示例值 |
| --- | --- |
| 机器人SSH用户 / IP | unitree / 192.168.123.164 |
| 机器人连接运动控制网络的网卡 | eth0 |
| PC连接机器人的网卡 / IPv4 | enp7s0 / 192.168.123.222 |
| PC Wi-Fi IPv4（填入PICO） | 用下面命令查看 |
| 左手 / 右手IP | 192.168.123.210 / 192.168.123.211 |
| 机器人项目目录 | /home/unitree/VLA/GR00T-WholeBodyControl |
| PC项目目录 | $HOME/sonic_v11/GR00T-WholeBodyControl |

### 0.1 机器人：查SSH用户名和机载IP

通过机器人本地终端，或已经能用的SSH连接进入机器人，执行：

```bash
whoami
hostname
ip -br -4 addr
```

`whoami`的输出是SSH用户名；`hostname`只是机器名称，不是用户名。`ip`输出每行是“网卡名、状态、IPv4/掩码”，例如：

```text
eth0  UP  192.168.123.164/24
```

记录PC能够连接的机载地址，去掉 `/24` 等掩码后缀。如果还不知道IP且无法SSH登录，先从机器人本地终端查看，或向设备管理员取得机载网络信息；不要把示例IP当成实际地址。多个网卡时，用下一步从PC验证所选地址。

### 0.2 PC：查机器人有线路由、网卡和PC有线IP

接好机器人网线，在PC执行；输入上一步确认的机载IP：

```bash
read -r -p "机器人机载IPv4地址: " ROBOT_IP
ip -br -4 addr
ip -4 route get "$ROBOT_IP"
```

路由结果示例：

```text
192.168.123.164 dev enp7s0 src 192.168.123.222
```

`dev`后面的名称是PC实际出站网卡，`src`后面的地址是PC有线IPv4。把两者分别记入参数表。若结果经过Wi-Fi、VPN或错误网卡，先让设备管理员修正有线连接/地址配置，再继续；不要仅因命令打印了结果就判定有线网络正常。

用第0.1步的用户名和IP验证SSH：

```bash
read -r -p "机器人SSH用户名: " ROBOT_USER
ssh "$ROBOT_USER@$ROBOT_IP" hostname
```

应显示第0.1步的机器人主机名。首次连接会询问信任主机，尚未配置免密时还会询问登录密码；第7步再设置免密。

### 0.3 PC：查Wi-Fi地址（头显连接用）

在PC执行：

```bash
for netdev in /sys/class/net/*; do
  if [ -d "$netdev/wireless" ]; then
    ip -br -4 addr show dev "${netdev##*/}"
  fi
done
```

例如输出 `wlp0s20f3 UP 10.0.0.25/24`，则PICO应用的PC Service填写 `10.0.0.25`，不是网卡名，也不是机器人的IP。没有输出或没有IPv4时，先将PC连入与PICO相同的Wi-Fi；也可在Ubuntu“设置→Wi-Fi→当前网络→详情”读取IPv4地址。

### 0.4 机器人：确认运动控制网卡

机器人SSH网卡不一定就是运动控制网卡。在机器人执行：

```bash
ip -br -4 addr
ip -br link
```

优先核对该机器人原有、已验证的SONIC启动命令中传入的网卡；没有记录时，从设备管理员取得运动控制器IP，再查询到该控制器的路由：

```bash
read -r -p "已确认的运动控制器IPv4地址: " CONTROLLER_IP
ip -4 route get "$CONTROLLER_IP"
```

结果中 `dev` 后的网卡作为候选，并核对设备接线/网络说明。第9步启动时必须实际收到身体状态，才能确认DDS链路可用。不能把PC有线网卡名填到机器人参数里，也不能在未核实的情况下统一填 `eth0`。

### 0.5 确认左右手IP及物理侧别

先查手部控制器配置页/设备网络记录，或原来可用的手桥接命令中的 `--left-ip`、`--right-ip`。Linux没有通用命令能自动推断哪一个IP对应左手。

PC可查看已经发现的邻居地址，辅助核对（不是完整设备清单）：

```bash
ip neigh show
```

取得候选IP后，检查本项目使用的TCP6000端口。此命令只连接端口，不发送手部动作：

```bash
read -r -p "待检查的手部IPv4地址: " HAND_IP
if timeout 3 bash -c 'exec 3<>/dev/tcp/"$1"/6000' _ "$HAND_IP"; then
  echo "TCP6000可连接；仍需核对手部型号和左右侧别"
else
  echo "连接失败，检查手部供电、地址、路由和端口"
fi
```

端口可连接不代表已经确认是Inspire或某一侧。最后在第8步分别发送一侧的小幅调整，通过真实响应核对左右；若左右反了，交换参数表中的两个IP。没有任何IP记录时，用手部厂商的网络配置工具读取，不盲猜地址或把其他设备当作手。

### 0.6 查项目路径及模型目录

新安装直接使用第2步创建的目录，不需要另找备份路径。复用已有安装时，在机器人和PC各进入自己的项目根目录执行：

```bash
pwd -P
git rev-parse --show-toplevel
```

记录绝对路径；机器人根目录对应启动参数 `--deploy-onboard-repo-root`，PC根目录用于后续命令中的 `cd`。模型相对路径不凭空猜测，第4步下载后应存在：

```text
gear_sonic_deploy/policy/sonic_v1_1/model_encoder.onnx
gear_sonic_deploy/policy/sonic_v1_1/model_decoder.onnx
gear_sonic_deploy/policy/sonic_v1_1/observation_config.yaml
```

### 0.7 将查到的值对应到命令

| 查到的值 | 后续替换位置 |
| --- | --- |
| 机器人SSH用户名 | 第7步SSH地址；非unitree时，第9步额外加 `--deploy-onboard-user 实际用户名` |
| 机载IP | `--camera-host`、`--deploy-onboard-host`及SSH地址 |
| 机器人运动控制网卡 | `--deploy-onboard-interface` |
| PC有线网卡 | 第9步 `--inspire-hand-network`；第8步网页直接在机器人控制手，不需要此参数 |
| PC有线IPv4 | `--offboard-zmq-host` |
| PC Wi-Fi IPv4 | 第6步头显应用的PC Service地址 |
| 左手、右手IP | 第8步 `--left`/`--right`；第9步 `--inspire-left-ip`/`--inspire-right-ip` |
| 机器人、PC项目路径 | 两端的 `cd`；机器人路径还用于 `--deploy-onboard-repo-root` |

**完成标准：**参数表填写完整，PC能SSH连接到正确机器人，有线路由及Wi-Fi地址区分清楚；手部物理侧别在第8步完成最终确认。

## 1. 机器人：检查系统基础

在机器人终端执行：

```bash
uname -m
cat /etc/os-release
cat /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core libnvinfer10 libnvinfer-dev
nvcc --version
```

支持的基础：AGX Orin使用JetPack 6＋TensorRT 10.7，AGX Thor使用JetPack 7.1＋TensorRT 10.13。TAR安装的TensorRT可能查不到系统包，以其安装目录的 `include/NvInferVersion.h` 为准。

**完成标准：**基础系统、CUDA及TensorRT已安装且版本匹配。本SOP从可用的机载系统开始，不包含刷机、CUDA或驱动更换；若缺失，先由设备维护者准备基础系统，再继续。不能拿不匹配的TensorRT engine直接运行。

## 2. 机器人和PC：安装工具，取得同一份代码

先分别在机器人和PC各执行一次：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs curl wget tmux nano openssh-client
git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

机器人执行：

```bash
mkdir -p /home/unitree/VLA
cd /home/unitree/VLA
git clone https://github.com/goldenstasteworriors/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git checkout 3c7bf4acfb2b93fd3168cf27143c95871f60ea11
git submodule update --init --recursive
git lfs pull
git rev-parse HEAD
```

PC执行：

```bash
mkdir -p "$HOME/sonic_v11"
cd "$HOME/sonic_v11"
git clone https://github.com/goldenstasteworriors/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git checkout 3c7bf4acfb2b93fd3168cf27143c95871f60ea11
git submodule update --init --recursive
git lfs pull
git rev-parse HEAD
```

**完成标准：**两端提交号均为上述固定提交，依赖仓库及LFS文件下载成功。这里只下载代码，使用HTTPS以免首次安装还要配置GitHub SSH密钥；若仓库访问受限，由管理员提供读取权限后重试。不要在已有同名项目目录上重跑clone，也不要为了继续安装删除旧目录；改用新的父目录，并同步修改参数表和后续路径。

## 3. 机器人：安装C++依赖并编译推理程序

先查已有TensorRT路径。在机器人执行：

```bash
printf '%s\n' "${TensorRT_ROOT:-尚未设置TensorRT_ROOT}"
dpkg-query -L libnvinfer-dev 2>/dev/null | grep '/NvInferVersion.h$'
```

第二条若显示 `/usr/include/.../NvInferVersion.h`，说明开发头文件来自系统包；用第1步的包版本确认版本。TAR安装则从原来的解压目录查找；忘记目录时可限定范围查：

```bash
find "$HOME" /opt /usr/local -maxdepth 4 -name NvInferVersion.h -print 2>/dev/null
```

例如找到 `/home/unitree/TensorRT-10.7.0.23/include/NvInferVersion.h`，则根目录是 `/home/unitree/TensorRT-10.7.0.23`，用它替换下方 `$HOME/TensorRT`。多个结果时按第1步要求核对版本，不按搜索顺序随意选择。

若TensorRT是TAR安装，先设置真实安装目录。下面示例仅适用于解压在机器人 `~/TensorRT` 的情况；JetPack系统包安装则跳过这两行，由环境脚本检测。

```bash
export TensorRT_ROOT="$HOME/TensorRT"
test -f "$TensorRT_ROOT/include/NvInferVersion.h"
```

在第1步已确认完整CUDA环境的机器人上执行：

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl/gear_sonic_deploy
bash scripts/install_deps.sh
source scripts/setup_env.sh
just build
test -x target/release/g1_deploy_onnx_ref && echo "部署程序已生成"
ldd target/release/g1_deploy_onnx_ref
```

依赖脚本在发现CUDA组件缺失时会尝试安装；基础组件缺失的机器不要继续跑此步骤，回到第1步处理。本流程使用ZMQ及Unitree DDS，无需额外配置ROS2。

**完成标准：**编译成功，显示“部署程序已生成”，且ldd输出没有 `not found`。若有缺失库，先修正依赖/库路径再继续；不能因为二进制文件存在就认定它能运行。本步不启动机器人控制。

## 4. 机器人：安装相机环境，下载v1.1模型

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl
bash install_scripts/install_camera_server.sh
```

脚本询问是否安装开机相机服务时，**输入n**；本文统一在第9步手动启动，避免重复服务。继续：

```bash
uv pip install --python .venv_camera/bin/python pyrealsense2 huggingface_hub
.venv_camera/bin/python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.serial_number) for d in rs.context().devices])"
.venv_camera/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/GEAR-SONIC', allow_patterns=['sonic_v1_1/model_encoder.onnx','sonic_v1_1/model_decoder.onnx','sonic_v1_1/observation_config.yaml'], local_dir='gear_sonic_deploy/policy')"
.venv_camera/bin/python download_from_hf.py
sha256sum gear_sonic_deploy/policy/sonic_v1_1/* gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
```

最后一个下载命令补齐planner等资源，实际控制在第9步显式选v1.1。模型encoder、decoder与观测配置三件套保持配套，不改网络和输入维度。将下载版本及SHA256保存到交付记录。

**完成标准：**打印出真实相机序列号，模型及planner存在且校验可读。若ARM64无法安装/导入pyrealsense2，需先完成匹配的RealSense SDK适配；不能跳过相机检查继续采集。

## 5. PC：安装PICO和数据采集环境

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
bash install_scripts/install_pico.sh
bash install_scripts/install_data_collection.sh
.venv_teleop/bin/python -c "import unitree_sdk2py; print('手控制依赖正常')"
.venv_teleop/bin/python gear_sonic/scripts/pico_manager_thread_server.py --help
.venv_data_collection/bin/python gear_sonic/scripts/run_data_exporter.py --help
.venv_data_collection/bin/python gear_sonic/scripts/launch_data_collection.py --help
```

**完成标准：**手控制依赖导入正常，启动器能显示帮助。安装脚本会重建对应venv，只在首次安装时执行。

## 6. PC和PICO：连接头显

**PC执行：**以下命令自动选择Ubuntu 22.04或24.04的x86_64安装包；其他PC平台不使用此命令。

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
. /etc/os-release
case "$VERSION_ID" in 22.04|24.04) ;; *) echo "本步骤仅支持Ubuntu 22.04/24.04"; exit 1 ;; esac
mkdir -p external_dependencies/downloads
wget -P external_dependencies/downloads "https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_${VERSION_ID}_amd64.deb"
sudo apt-get install "./external_dependencies/downloads/XRoboToolkit_PC_Service_1.0.0_ubuntu_${VERSION_ID}_amd64.deb"
```

从PC桌面应用菜单打开XRoboToolkit PC Service，保持运行。

**头显依次操作：**

1. 双脚踝追踪器指示灯朝上，不被衣物遮挡；在Developer设置关闭Safeguard。
2. Motion Tracker应用清除旧配对，长按两只追踪器按钮约6秒进入红蓝闪烁配对，完成配对。
3. 按Calibrate：先站直、手柄垂于两侧，再低头让头显识别脚踝追踪器。
4. 头显浏览器打开[APK下载地址](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk)，安装后从应用库Unknown打开。
5. PICO和PC接入同一Wi-Fi。在头显应用PC Service处填第0步记录的PC Wi-Fi地址；勾选Head、Controller，Data/Control选Send，Motion Tracker选Full body。

**完成标准：**头显连接状态为WORKING，手柄和全身追踪正常。

## 7. PC：检查到机器人的连接

```bash
ssh unitree@192.168.123.164 hostname
```

确保PC可以免密SSH登录，供启动器使用。先在PC查看是否已有公钥：

```bash
ls "$HOME/.ssh/id_ed25519.pub" "$HOME/.ssh/id_rsa.pub" 2>/dev/null
```

如果已经有默认公钥，跳过下面的生成命令；如果一个也没有，则执行：

```bash
ssh-keygen -t ed25519
```

保存位置按回车使用默认值；若提示文件已存在，不覆盖，先核对现有密钥。按终端提示设置密钥口令，使用有口令的密钥时先将其加载到当前桌面的ssh-agent，确保下方BatchMode检查通过。随后把PC公钥安装到机器人（替换用户名/IP，输入机器人登录密码）：

```bash
ssh-copy-id unitree@192.168.123.164
ssh -o BatchMode=yes unitree@192.168.123.164 hostname
```

如果密钥有口令且BatchMode失败，在PC当前终端加载密钥后重试；下面以默认ed25519密钥为例，rsa密钥改用实际文件名：

```bash
eval "$(ssh-agent -s)"
ssh-add "$HOME/.ssh/id_ed25519"
ssh -o BatchMode=yes unitree@192.168.123.164 hostname
```

后续采集从这个已加载密钥的PC终端启动，使新建tmux会话能够继承agent。每次重新登录PC后先重做BatchMode检查。

**完成标准：**第二条命令直接显示机器人主机名，不询问登录密码。启用防火墙时还需允许：机器人→PC TCP5556（PICO输入），PC→机器人TCP5557（状态反馈）、5555（相机）。

## 8. 机器人启动手控网页，PC调节并保存配置

调节前退出采集和手桥接。本步骤使用机器人备份中的原工具，实际文件名是 **ctlhand.py**（不是ctrl_hand.py）。第2步固定版本已经包含该文件，不需要另找外部脚本，也不需要再安装整个locodemo项目。

**8.1 机器人：首次安装独立手控环境。**

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl
test -f tools/inspire_hand_web/ctlhand.py
uv venv .venv_hand_web --python 3.10
uv pip install --python .venv_hand_web/bin/python -r tools/inspire_hand_web/requirements.txt
.venv_hand_web/bin/python tools/inspire_hand_web/ctlhand.py --help
```

只在首次安装执行uv建环境；后续从8.2开始。原文件保持不变，直接通过Modbus控制手；此时不要另外开DDS手桥接。

**8.2 机器人终端：**使用第0步确认的左右手IP启动，并保持运行。

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl
.venv_hand_web/bin/python tools/inspire_hand_web/ctlhand.py --left 192.168.123.210 --right 192.168.123.211 --host 127.0.0.1 --web-port 8080
```

**8.3 PC终端：**将SSH用户和机器人IP替换为实际值，建立隧道，保持运行。

```bash
ssh -N -L 8080:127.0.0.1:8080 unitree@192.168.123.164
```

**8.4 PC浏览器打开：**

```text
http://localhost:8080/
```

分别选择左手、右手，使用“六电机”目标百分比调节并发送。记录张开、抓握各6个**目标值**：网页百分比0～100需除以100；原始angle目标0～1000需除以1000，转换为采集配置的0～1。例如50%对应0.5。不要把受负载影响的实际角度反馈当成目标值。

六电机顺序：**小指、无名指、中指、食指、拇指弯曲、拇指旋转**；配置值域0～1，通常0闭合、1张开。

打开PC项目中的配置文件：

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
nano gear_sonic/config/data_collection/inspire_hand_pose.json
```

用实测值替换下面的示例，保存退出（nano：Ctrl+O、回车、Ctrl+X）：

```json
{
  "open": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
  "grasp": [0.5, 0.5, 0.5, 0.5, 1.0, 0.0]
}
```

**网页不会自动保存该JSON。** 当前配置对双手共用一套开合目标；左右若必须用不同目标，需要先扩展配置和桥接后再交付。记录任务、日期和调节值。

保存后在PC验证JSON格式：

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
.venv_data_collection/bin/python -m json.tool gear_sonic/config/data_collection/inspire_hand_pose.json
```

应正常打印配置而不报错；仍需人工确认open、grasp各有6个0～1数值。采集桥接在启动时读取该配置，修改后要停止旧桥接并重新启动第9.3步，不能指望旧进程自动刷新。

**完成标准：**网页发送后真实手响应正常，JSON已保存。完成后在机器人手控终端及PC隧道终端分别Ctrl+C退出，防止与采集手桥接冲突。

## 9. 每次使用：按顺序启动相机、检查图像、采集

**9.1 机器人终端：**接好相机并启动，保持运行。已有同端口相机服务则复用，不重复启动。

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl
.venv_camera/bin/python -m gear_sonic.camera.composed_camera --ego-view-camera realsense --port 5555
```

**9.2 PC终端：**先打开PC Service并连接PICO，再检查图像；确认画面连续更新后按q退出查看器。

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
.venv_data_collection/bin/python gear_sonic/scripts/run_camera_viewer.py --camera-host 192.168.123.164 --camera-port 5555
```

**9.3 PC终端：**替换第0步参数、任务提示和新的dataset-name后执行。启动器会启动机器人推理及PC手桥接，不再另外开手桥接。

每次启动前，确认第8步手控网页已退出、第7步免密检查通过，并且第9.1步相机仍在运行。重新运行启动器会关闭同名旧tmux会话并重启桥接，已有采集应先正常保存并停止。

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
.venv_data_collection/bin/python gear_sonic/scripts/launch_data_collection.py \
  --camera-host 192.168.123.164 \
  --deploy-onboard --deploy-onboard-host 192.168.123.164 \
  --deploy-onboard-repo-root /home/unitree/VLA/GR00T-WholeBodyControl \
  --deploy-onboard-interface eth0 --offboard-zmq-host 192.168.123.222 \
  --deploy-checkpoint policy/sonic_v1_1/model \
  --deploy-obs-config policy/sonic_v1_1/observation_config.yaml \
  --deploy-planner planner/target_vel/V2/planner_sonic.onnx \
  --inspire-hand-bridge --inspire-hand-network enp7s0 \
  --inspire-left-ip 192.168.123.210 --inspire-right-ip 192.168.123.211 \
  --inspire-hand-pose-config gear_sonic/config/data_collection/inspire_hand_pose.json \
  --task-prompt "pick up the box" --dataset-name pick_box_v11_20261110_01 \
  --enable-sonic-publish --no-pico-vis-smpl --no-pico-vis-vr3pt \
  --profile-timing
```

现场确认机器人已准备好后，在deploy窗口按y允许部署。

启动后进入tmux：`data_collection`窗口左上是机器人C++推理，左下是PICO，右上是数据导出，右下是相机。鼠标点击对应pane即可输入；也可先按Ctrl+B松开，再按方向键切换pane。Ctrl+B松开再按W选择窗口，进入 `inspire_hand` 查看手桥接。出现“All components launched”只说明命令已发送；各pane不能有Traceback、持续连接失败或进程退出。

若只是终端断开，在PC重新进入（不会重新启动控制）：

```bash
tmux attach -t sonic_data_collection
```

进入控制和录制时依次操作：

1. 站直、双脚并拢、大臂贴体、小臂向前弯曲90度、掌心相对。
2. A+B+X+Y启动策略并校准；等待稳定，对齐手臂，A+X进入POSE遥操作。
3. 焦点放到数据导出pane：c开始录制，再按c结束保存；录制中x丢弃当前段。
4. A+X返回PLANNER；A+B+X+Y停止策略，C++终端O也是停止入口。PLANNER不等于停机，确认停止后再退出程序；断开tmux不代表停机。

每批使用新dataset-name，不加覆盖已有数据的参数。

## 10. PC：检查数据后交付

打开项目输出目录（将dataset-name替换为本次名称）：

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
ls outputs/pick_box_v11_20261110_01
```

**完成标准：**meta、data、videos均存在；视频可播放，帧数与episode对应；state/action为41维且顺序是身体29、左手6、右手6，motion token、teleop和modality信息保留。50Hz记录不代表相机有50张独立曝光。

交付代码固定提交、v1.1三件套及planner、模型版本/SHA256、手姿态JSON、网络参数表、依赖版本和本SOP。首次安装必须实际录制并检查一条短episode后，才算整套验收完成。

## 工具来源

机载网页工具从本机机器人备份原样复制，已收入交付代码，依赖及详细用法见tools/inspire_hand_web/README.md。

```text
原备份：/home/ykj/Downloads/data_collection/VLA/locodemo/ctlhand.py
交付位置：tools/inspire_hand_web/ctlhand.py
SHA256：86538b20698e205aabbb29865864b56a2e42630fcf500c620d1462e490494d3f
```

## 参考与验证范围

官方资料仅作来源参考，无需再跟随其安装流程：[机载部署](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html)、[模型](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/download_models.html)、[PICO配置](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html)、[遥操作](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vr_wholebody_teleop.html)。

2026-11-10：命令参数、数据维度及Shell/JSON语法按本项目源码核对；手控工具通过本地HTTP启动、模拟手对象接口及锁定版本pymodbus兼容性验证，未向机器人发送请求。机器人连接超时，尚未完成干净环境安装和真机验收。适用范围是第1步已具备匹配基础系统的设备；未找到可确认版本的完整机器人环境归档，不将现有代码备份称为已验证镜像。
